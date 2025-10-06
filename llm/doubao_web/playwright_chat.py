from playwright.sync_api import sync_playwright
import time
import os
import sqlite3
import platform
import shutil
from typing import Optional
import sys
import random
import re

def _apply_stealth(page):
    """为页面注入隐身脚本，降低自动化指纹暴露概率。"""
    try:
        page.set_extra_http_headers({
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
        })
    except Exception:
        pass
    try:
        page.add_init_script(r"""
            // webdriver 痕迹
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

            // window.chrome 与插件
            window.chrome = window.chrome || { runtime: {} };
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    { name: 'Chrome PDF Plugin' },
                    { name: 'Chrome PDF Viewer' },
                    { name: 'Native Client' }
                ]
            });

            // 语言与平台
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN','zh','en-US','en'] });
            Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });

            // userAgent 去掉 HeadlessChrome
            try {
                const ua = navigator.userAgent.replace('HeadlessChrome', 'Chrome');
                Object.defineProperty(navigator, 'userAgent', { get: () => ua });
            } catch (e) {}

            // 性能参数
            Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
            Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });

            // permissions 拦截
            try {
                const originalQuery = navigator.permissions && navigator.permissions.query;
                if (originalQuery) {
                    navigator.permissions.query = (parameters) => (
                        parameters && parameters.name === 'notifications'
                            ? Promise.resolve({ state: Notification.permission })
                            : originalQuery(parameters)
                    );
                }
            } catch (e) {}

            // WebGL vendor/renderer
            try {
                const getParameter = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(param) {
                    if (param === 37445) return 'Intel Inc.'; // UNMASKED_VENDOR_WEBGL
                    if (param === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
                    return getParameter.call(this, param);
                };
            } catch (e) {}

            // outer/inner 尺寸一致
            try {
                Object.defineProperty(window, 'outerWidth', { get: () => window.innerWidth });
                Object.defineProperty(window, 'outerHeight', { get: () => window.innerHeight });
            } catch (e) {}
        """)
    except Exception:
        pass

"""
基于 Playwright 的豆包网页自动化（持久化会话方案）
用法：
  1) 首次人工登录并保存会话：python playwright_chat.py bootstrap
  2) 复用会话聊天：           python playwright_chat.py chat "你的问题"
"""

# 统一选择器与常量，便于维护与复用
INPUT_HINT_SELECTORS = [
    "textarea[placeholder*='发消息']",
    "textarea[placeholder*='输入']",
    "textarea[placeholder*='/ 选择技能']",
]
INPUT_GENERAL_SELECTORS = [
    "textarea",
    "[contenteditable='true']",
    ".ProseMirror[contenteditable='true']",
    "div[role='textbox']",
    "div[data-placeholder]",
    "div[placeholder]",
    ".chat-input textarea",
    ".chat-input [contenteditable='true']",
]
SEND_BUTTON_SELECTORS = [
    "button:has-text('发送')",
    "button[aria-label='发送']",
    "button[type='submit']",
]
RESPONSE_SELECTORS = [
    ".message-list .assistant .message-content",
    ".message-item .markdown-content",
    ".message-item .reply-content",
    ".message-container .content",
    ".markdown-body",
    "article",
]
CAPTCHA_SELECTORS = [
    "text=验证", "text=安全验证", "text=请完成验证",
    "iframe[src*='geetest']", "iframe[src*='captcha']", "iframe[src*='hcaptcha']",
]
COMPLETE_TOOLBAR_TEXTS = ["编辑", "分享"]

def _wait_page_ready(page, timeout_ms: int = 15000):
    """等待页面加载完成且聊天输入框出现，避免过早输入触发风控。
    1) 等待 DOM 内容加载
    2) 尝试等待网络空闲（若不可用则忽略）
    3) 等待任一输入框提示或可编辑区域可见
    """
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass
    try:
        # 部分站点可能不支持/不稳定，故捕获异常
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass
    # 统一等待输入框准备就绪
    try:
        script = """
            sel => {
              const list = JSON.parse(sel);
              for (const s of list) {
                const el = document.querySelector(s);
                if (el && el.offsetParent !== null) return true;
              }
              // 可编辑区域也算
              const ce = document.querySelector("[contenteditable='true']");
              return !!(ce && ce.offsetParent !== null);
            }
        """
        page.wait_for_function(script, json.dumps(INPUT_HINT_SELECTORS + INPUT_GENERAL_SELECTORS), timeout=timeout_ms)
    except Exception:
        # 若失败，不阻塞；后续 _find_input 会再次检查
        pass

def _find_input(page):
    """在主文档与所有 iframe 中查找聊天输入框，返回第一个可用定位器。"""
    def _scan(fr):
        for sel in INPUT_HINT_SELECTORS:
            loc = fr.locator(sel)
            if loc.count() > 0:
                return loc.first
        for sel in INPUT_GENERAL_SELECTORS:
            loc = fr.locator(sel)
            if loc.count() > 0:
                return loc.first
        return None

    # 主文档优先
    loc = _scan(page)
    if loc:
        return loc
    # 跨 iframe 搜索
    for fr in page.frames:
        try:
            loc = _scan(fr)
            if loc:
                return loc
        except Exception:
            continue
    # 等待候选选择器一段时间（页面可能仍在渲染）
    try:
        candidates = ", ".join(INPUT_HINT_SELECTORS + INPUT_GENERAL_SELECTORS)
        page.wait_for_selector(candidates, timeout=15000)
        loc = page.locator(candidates)
        if loc.count() > 0:
            return loc.first
    except Exception:
        pass
    return None

def _human_type(page, target_loc, text: str):
    """类人逐字输入：在定位器上打字，带轻微停顿与抖动。"""
    base_ms = random.randint(20, 60)
    for ch in text:
        delay_ms = max(10, base_ms + random.randint(-10, 10))
        try:
            target_loc.type(ch, delay=delay_ms)
        except Exception:
            page.keyboard.type(ch, delay=delay_ms)
        time.sleep(random.uniform(0.005, 0.03))
        if random.random() < 0.02:
            time.sleep(random.uniform(0.08, 0.25))

def _user_message_appeared(page, prompt: str) -> bool:
    """检测页面是否出现了当前用户消息（用前缀文本做包含匹配）。"""
    try:
        if page.get_by_text(prompt[:20]).count() > 0:
            return True
    except Exception:
        pass
    return False

def _send_message(page, input_loc, prompt: str) -> bool:
    """聚焦输入框并发送消息，逐字输入，返回是否成功（不强制等待用户消息回显）。"""
    try:
        input_loc.scroll_into_view_if_needed(timeout=5000)
    except Exception:
        pass
    try:
        page.mouse.move(random.randint(100, 300), random.randint(200, 400))
    except Exception:
        pass
    time.sleep(random.uniform(0.2, 0.6))
    input_loc.click(timeout=10000)
    try:
        input_loc.focus()
    except Exception:
        pass
    # 始终使用逐字输入，模拟人类打字（参考 playwright_chat_1006.py）
    _human_type(page, input_loc, prompt)

    sent = False
    try:
        time.sleep(random.uniform(0.15, 0.35))
        page.keyboard.press("Enter")
        sent = True
    except Exception:
        sent = False
    if not sent:
        for send_sel in SEND_BUTTON_SELECTORS:
            try:
                page.locator(send_sel).first.click()
                sent = True
                break
            except Exception:
                continue

    # 非阻塞降级：尝试不同快捷键，但不强制等待用户消息出现
    if not sent:
        for key in ["Control+Enter", "Meta+Enter", "Shift+Enter", "Enter"]:
            try:
                page.keyboard.press(key)
                sent = True
                break
            except Exception:
                continue
    if not sent:
        try:
            input_loc.locator("xpath=../..//button").first.click(timeout=3000)
            sent = True
        except Exception:
            pass
    return sent

def _extract_reply(page, prompt: str, wait_timeout_sec: int = 60, headless: bool = True) -> Optional[str]:
    """尽快返回首段回复：优先靠近用户消息的块，快速轮询，出现文本即返回。"""
    # 页面关闭保护：若页面已关闭，立即返回
    try:
        if hasattr(page, "is_closed") and page.is_closed():
            return None
    except Exception:
        pass

    # 快速早返回：等待任一回复选择器出现（JS 端一次性检测），避免长时间 Python 轮询
    try:
        js = """
        sels => {
          const list = JSON.parse(sels);
          for (const s of list) {
            const el = document.querySelector(s);
            if (el && el.offsetParent !== null) return true;
          }
          return false;
        }
        """
        ok = page.wait_for_function(js, json.dumps(RESPONSE_SELECTORS), timeout=int(min(wait_timeout_sec, 20) * 1000))
        if ok:
            # 立即抓取最后一个匹配块的文本
            for sel in RESPONSE_SELECTORS:
                try:
                    loc = page.locator(sel)
                    cnt = loc.count()
                    if cnt > 0:
                        text = loc.nth(cnt - 1).inner_text()
                        if text and text.strip():
                            return text.strip()
                except Exception:
                    continue
    except Exception:
        pass
    # 轻量验证码检测（仅提示，不阻塞）
    try:
        for sel in CAPTCHA_SELECTORS:
            if page.locator(sel).count() > 0:
                if not headless:
                    print("检测到可能的验证弹窗，请完成后继续...")
                break
    except Exception:
        pass

    # 1) 优先：基于用户消息的最近兄弟容器，快速轮询返回首段文本
    try:
        anchor_loc = page.get_by_text(prompt[:20])
        if anchor_loc.count() > 0:
            try:
                anchor = anchor_loc.last
            except Exception:
                anchor = None
            if anchor:
                candidate = anchor.locator(
                    "xpath=ancestor::*[self::div or self::section or self::article][1]/following-sibling::*[self::div or self::section or self::article][1]"
                )
                if candidate.count() > 0:
                    start = time.time()
                    while time.time() - start < wait_timeout_sec:
                        # 页面关闭保护
                        try:
                            if hasattr(page, "is_closed") and page.is_closed():
                                return None
                        except Exception:
                            return None
                        try:
                            for sub in [".markdown", ".markdown-body", "article", "div:has(p)", "div:has(pre)"]:
                                loc2 = candidate.locator(sub)
                                if loc2.count() > 0:
                                    text = loc2.first.inner_text()
                                    if text and text.strip():
                                        return text.strip()
                            txt = candidate.inner_text()
                            if txt and txt.strip():
                                return txt.strip()
                        except Exception:
                            pass
                        time.sleep(0.1)
    except Exception:
        pass

    # 2) 常规：从已知回复选择器中取最后一个，快速轮询；若出现工具条则取其上方内容
    start = time.time()
    while time.time() - start < wait_timeout_sec:
        # 页面关闭保护
        try:
            if hasattr(page, "is_closed") and page.is_closed():
                return None
        except Exception:
            return None
        # 工具条（编辑/分享）出现时，优先取其上方的内容块，不阻塞等待
        try:
            for t in COMPLETE_TOOLBAR_TEXTS:
                loc = page.get_by_text(t)
                if loc.count() > 0:
                    try:
                        content_node = loc.last.locator("xpath=preceding-sibling::*[1]")
                        if content_node.count() > 0:
                            text = content_node.inner_text()
                            if text and text.strip():
                                return text.strip()
                    except Exception:
                        continue
        except Exception:
            pass
        try:
            for sel in RESPONSE_SELECTORS:
                loc = page.locator(sel)
                cnt = 0
                try:
                    cnt = loc.count()
                except Exception:
                    # 页面或上下文可能已关闭，结束提取
                    return None
                if cnt > 0:
                    try:
                        text = loc.nth(cnt - 1).inner_text()
                        if text and text.strip():
                            return text.strip()
                    except Exception:
                        continue
        except Exception:
            # 页面或上下文可能已关闭
            return None
        # 评估回退：一次性扫描页面文本，找到最可能的回复块（shadow DOM 的场景）
        try:
            txt = page.evaluate(
                """
                () => {
                  const sels = [
                    '.message-list .assistant .message-content',
                    '.message-item .markdown-content',
                    '.message-item .reply-content',
                    '.message-container .content',
                    '.markdown-body',
                    'article'
                  ];
                  for (const s of sels) {
                    const el = document.querySelector(s);
                    if (el) {
                      const t = (el.innerText || '').trim();
                      if (t) return t;
                    }
                  }
                  const all = Array.from(document.querySelectorAll('*'));
                  for (const el of all) {
                    const text = (el.textContent || '').trim();
                    if (text && (text.includes('编辑') || text.includes('分享'))) {
                      const prev = el.previousElementSibling;
                      if (prev) {
                        const pt = (prev.innerText || '').trim();
                        if (pt) return pt;
                      }
                    }
                  }
                  return '';
                }
                """
            )
            if txt and txt.strip():
                return txt.strip()
        except Exception:
            pass
        time.sleep(0.1)

    # 3) 终极回退：以用户消息为切分点，从 body 可见文本中截取
    try:
        body_text = page.locator("body").inner_text()
        if body_text:
            idx = body_text.rfind(prompt[:20])
            if idx != -1:
                tail = body_text[idx + len(prompt[:20]):]
                cleaned = tail.strip()
                cleaned = re.sub(r"[\s\u00A0]+", " ", cleaned)
                if cleaned:
                    return cleaned
    except Exception:
        pass

    return None

def get_chrome_cookies_path():
    """获取本地Chrome的Cookie数据库路径"""
    system = platform.system()
    home = os.path.expanduser("~")
    
    if system == "Windows":
        return os.path.join(home, "AppData", "Local", "Google", "Chrome", "User Data", "Default", "Network", "Cookies")
    elif system == "Darwin":  # macOS
        return os.path.join(home, "Library", "Application Support", "Google", "Chrome", "Default", "Network", "Cookies")
    elif system == "Linux":
        return os.path.join(home, ".config", "google-chrome", "Default", "Network", "Cookies")
    else:
        raise OSError(f"不支持的操作系统: {system}")

def copy_cookie_file():
    """复制Chrome Cookie文件到临时位置（解决文件锁定和权限问题）"""
    try:
        chrome_cookies_path = get_chrome_cookies_path()
        
        # 检查源文件是否存在
        if not os.path.exists(chrome_cookies_path):
            print(f"未找到Chrome Cookie文件: {chrome_cookies_path}")
            return None
            
        # 创建临时目录
        temp_dir = os.path.join(os.getcwd(), "temp_cookies")
        os.makedirs(temp_dir, exist_ok=True)
        temp_cookie_path = os.path.join(temp_dir, "Cookies")
        
        # 尝试复制文件，处理权限问题
        try:
            # 先删除可能存在的旧文件
            if os.path.exists(temp_cookie_path):
                os.remove(temp_cookie_path)
            shutil.copy2(chrome_cookies_path, temp_cookie_path)
            print(f"已复制Cookie文件到临时位置: {temp_cookie_path}")
            return temp_cookie_path
        except PermissionError:
            print("无法访问Cookie文件，可能是因为Chrome正在运行中")
            print("请关闭所有Chrome窗口后重试")
            return None
        except Exception as e:
            print(f"复制Cookie文件失败: {str(e)}")
            return None
            
    except Exception as e:
        print(f"处理Cookie文件时出错: {str(e)}")
        return None

def load_cookies_to_browser(context, domain="doubao.com"):
    """将Cookie加载到Playwright浏览器上下文"""
    try:
        # 获取临时Cookie文件路径
        temp_cookie_path = copy_cookie_file()
        if not temp_cookie_path:
            return False
        
        # 连接到Cookie数据库
        conn = sqlite3.connect(temp_cookie_path)
        cursor = conn.cursor()
        
        # 查询指定域名的Cookie
        cursor.execute("""
            SELECT name, value, path, host_key, expires_utc, is_secure, is_httponly 
            FROM cookies 
            WHERE host_key LIKE ?
        """, (f"%{domain}%",))
        
        cookies = []
        for row in cursor.fetchall():
            name, value, path, host_key, expires_utc, is_secure, is_httponly = row

            # 转换有效期：Chrome 使用自 1601-01-01 的微秒时间戳
            # Playwright 期望的是 Unix epoch（1970-01-01）秒
            expires = None
            try:
                if expires_utc and int(expires_utc) > 0:
                    # 先转秒，再减去 1601 与 1970 的秒差
                    expires_unix = int(expires_utc // 1000000 - 11644473600)
                    if expires_unix > 0:
                        expires = expires_unix
            except Exception:
                expires = None

            cookies.append({
                "name": name,
                "value": value or "",
                "path": path or "/",
                "domain": host_key,
                "expires": expires,
                "secure": bool(is_secure),
                "httpOnly": bool(is_httponly),
                "sameSite": "Lax",
            })
        
        conn.close()
        
        # 添加Cookie到浏览器上下文
        if cookies:
            context.add_cookies(cookies)
            print(f"已加载 {len(cookies)} 个豆包相关Cookie")
            return True
        else:
            print("未找到豆包相关的Cookie，请先在Chrome中登录豆包")
            return False
            
    except Exception as e:
        print(f"加载Cookie时出错: {str(e)}")
        return False

def get_doubao_response(prompt: str, headless: bool = False) -> Optional[str]:
    """使用本地Chrome的Cookie与豆包交互"""
    try:
        with sync_playwright() as p:
            user_data_dir = os.path.join(os.getcwd(), "temp_profile")
            os.makedirs(user_data_dir, exist_ok=True)
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            )

            if not load_cookies_to_browser(context):
                context.close()
                return None

            page = context.pages[0] if context.pages else context.new_page()
            _apply_stealth(page)
            page.goto("https://www.doubao.com/", wait_until="domcontentloaded")
            _wait_page_ready(page, timeout_ms=15000)
            print(f"当前页面: {page.url}")

            input_loc = _find_input(page)
            if not input_loc:
                print("未找到输入框，可能尚未登录或页面结构变化。")
                context.close()
                return None

            if not _send_message(page, input_loc, prompt):
                print("消息发送失败或未出现用户消息。")
                context.close()
                return None

            reply = _extract_reply(page, prompt, wait_timeout_sec=60, headless=headless)

            try:
                storage_state_path = os.path.join(user_data_dir, "doubao_state.json")
                context.storage_state(path=storage_state_path)
            except Exception:
                pass
            context.close()
            return reply

    except Exception as e:
        print(f"操作出错: {str(e)}")
        return None

# === 持久化登录与复用 ===
def save_persistent_login_state(
    url: str = "https://www.doubao.com/",
    user_data_dir: str = os.path.join(os.getcwd(), "temp_profile"),
    headless: bool = False,
    check_selector: str = "textarea[placeholder='发消息或输入 / 选择技能']",
    timeout_sec: int = 180,
) -> bool:
    """
    启动持久化浏览器上下文，人工完成登录后，保留会话到 user_data_dir。
    后续使用同一个 user_data_dir 即可免登录。
    """
    try:
        with sync_playwright() as p:
            os.makedirs(user_data_dir, exist_ok=True)
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                channel="chrome",  # 使用已安装的 Chrome，降低自动化指纹
                args=["--disable-blink-features=AutomationControlled"],
                viewport=None,
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            # 注入隐身与真实请求头
            _apply_stealth(page)
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            _wait_page_ready(page, timeout_ms=15000)
            print("请在打开的浏览器中完成豆包登录。登录完成后，请手动关闭该窗口以结束并保存会话。")
            print("我会在后台检测：1) 聊天输入框出现表示登录完成；2) 页面关闭表示你已完成。")

            # 监听用户手动关闭页面
            closed = {"flag": False}
            def _on_close(*_):
                closed["flag"] = True
            try:
                page.on("close", _on_close)
            except Exception:
                pass

            deadline = time.time() + timeout_sec
            login_confirmed = False
            # 循环等待：先确认登录，再等待你关闭页面
            while time.time() < deadline and not closed["flag"]:
                try:
                    if not login_confirmed:
                        if page.locator(check_selector).count() > 0:
                            login_confirmed = True
                            print("已检测到聊天输入框，登录完成。可关闭窗口以保存会话。")
                            try:
                                ctx.storage_state(path=os.path.join(user_data_dir, "doubao_state.json"))
                            except Exception:
                                pass
                        elif page.locator("[contenteditable='true']").count() > 0:
                            login_confirmed = True
                            print("已检测到可编辑输入区域，登录完成。可关闭窗口以保存会话。")
                            try:
                                ctx.storage_state(path=os.path.join(user_data_dir, "doubao_state.json"))
                            except Exception:
                                pass
                except Exception:
                    pass
                time.sleep(1)

            # 根据关闭或登录确认结果返回
            if closed["flag"]:
                print(f"已检测到窗口关闭，登录会话已保存：{user_data_dir}")
                try:
                    ctx.close()
                except Exception:
                    pass
                return True
            elif login_confirmed:
                print(f"登录已确认（未检测到窗口关闭），已保留会话目录：{user_data_dir}")
                try:
                    ctx.close()
                except Exception:
                    pass
                return True
            else:
                print("未能在超时前确认登录，也未检测到窗口关闭，请重试。")
                try:
                    ctx.close()
                except Exception:
                    pass
                return False
    except Exception as e:
        print(f"保存登录状态出错: {e}")
        return False


def chat_once_persistent(
    prompt: str,
    url: str = "https://www.doubao.com/",
    user_data_dir: str = os.path.join(os.getcwd(), "temp_profile"),
    headless: bool = True,
    wait_timeout_sec: int = 60,
) -> Optional[str]:
    """
    使用持久化上下文直接复用登录态进行一次聊天，并返回回复文本。
    """
    try:
        with sync_playwright() as p:
            os.makedirs(user_data_dir, exist_ok=True)
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"],
                viewport=None,
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            _apply_stealth(page)
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            _wait_page_ready(page, timeout_ms=15000)

            input_loc = _find_input(page)
            if not input_loc:
                print("未找到输入框，可能尚未登录或页面结构变化。")
                try:
                    ctx.close()
                except Exception:
                    pass
                return None

            if not _send_message(page, input_loc, prompt):
                print("消息发送失败或未出现用户消息。")
                try:
                    ctx.close()
                except Exception:
                    pass
                return None

            reply_text = _extract_reply(page, prompt, wait_timeout_sec=wait_timeout_sec, headless=headless)
            try:
                ctx.close()
            except Exception:
                pass
            return reply_text
    except Exception as e:
        print(f"持久化聊天出错: {e}")
        return None

if __name__ == "__main__":
    # 用法：
    # 1) 首次手动登录保存会话：python playwright_chat.py bootstrap
    # 2) 之后直接复用会话聊天：python playwright_chat.py chat "你的问题"
    args = sys.argv[1:]
    if args and args[0] == "bootstrap":
        ok = save_persistent_login_state(headless=False)
        print("bootstrap:", "成功" if ok else "失败")
        # 在窗口关闭并成功保存会话后立刻退出程序（返回适当退出码）
        try:
            sys.exit(0 if ok else 1)
        except SystemExit:
            pass
    elif args and args[0] == "chat":
        prompt = " ".join(args[1:]) or "请解释什么是人工智能"
        # 默认缩短等待时间以加快返回，可通过环境变量或参数扩展
        resp = chat_once_persistent(prompt, headless=False, wait_timeout_sec=30)
        if resp:
            print(f"回复: {resp}")
        else:
            print("未能获取到回复")
    else:
        print("用法:\n  python playwright_chat.py bootstrap\n  python playwright_chat.py chat \"你的问题\"\n")
    