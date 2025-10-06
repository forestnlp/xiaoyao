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
            # 使用持久化上下文，稳定复用登录态
            user_data_dir = os.path.join(os.getcwd(), "temp_profile")
            os.makedirs(user_data_dir, exist_ok=True)
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            )
            
            # 加载Cookie
            if not load_cookies_to_browser(context):
                context.close()
                return None
            
            page = context.pages[0] if context.pages else context.new_page()
            
            # 导航到豆包网页
            page.goto("https://www.doubao.com/", wait_until="networkidle")
            print(f"当前页面: {page.url}")

            # 等待聊天输入框或回退到 contenteditable 通用输入
            input_selector = "textarea[placeholder='发消息或输入 / 选择技能']"
            input_loc = page.locator(input_selector)
            try:
                if input_loc.count() == 0:
                    # 回退：使用 contenteditable
                    fallback_input = page.locator("[contenteditable='true']")
                    fallback_input.first.wait_for(timeout=10000)
                    input_loc = fallback_input.first
                    input_selector_for_press = "[contenteditable='true']"
                else:
                    input_loc.first.wait_for(timeout=10000)
                    input_selector_for_press = input_selector
                print("已使用Cookie登录豆包")
            except Exception:
                print("登录失败或未发现输入框，请检查是否已在Chrome中登录豆包")
                context.close()
                return None
            
            # 输入并发送消息（兼容 contenteditable 与标准输入框）
            inp = input_loc
            try:
                attr = (inp.get_attribute("contenteditable") or "").lower()
            except Exception:
                attr = ""

            if attr == "true":
                inp.click()
                page.keyboard.type(prompt, delay=10)
            else:
                inp.fill(prompt)

            # 首选回车发送，失败则尝试常见发送按钮
            sent = False
            try:
                page.press(input_selector_for_press, "Enter")
                sent = True
            except Exception:
                pass

            if not sent:
                for send_sel in [
                    "button:has-text('发送')",
                    "button[aria-label='发送']",
                    "button[type='submit']",
                ]:
                    try:
                        page.locator(send_sel).first.click()
                        sent = True
                        break
                    except Exception:
                        continue
            
            # 等待回复
            time.sleep(2)
            response_selectors = [
                ".message-list .assistant .message-content",
                ".message-item .markdown-content",
                ".message-item .reply-content",
                ".message-container .content",
                ".markdown-body",
                "article",
            ]
            
            start_time = time.time()
            message_element = None
            
            while True:
                # 检查加载状态
                loading = page.query_selector_all(".loading, .typing")
                if not loading:
                    for selector in response_selectors:
                        loc = page.locator(selector)
                        count = loc.count()
                        if count > 0:
                            try:
                                message_element = loc.nth(count - 1).element_handle()
                            except Exception:
                                message_element = None
                            if message_element:
                                break
                    if message_element:
                        break
                
                if time.time() - start_time > 60:
                    print("等待回复超时")
                    context.close()
                    return None
                
                time.sleep(1)
            
            if not message_element:
                print("未找到回复内容")
                context.close()
                return None
            
            response = message_element.text_content()
            # 持久化当前会话状态，便于下次免登录
            try:
                storage_state_path = os.path.join(user_data_dir, "doubao_state.json")
                context.storage_state(path=storage_state_path)
            except Exception:
                pass
            context.close()
            
            return response.strip() if response else None
            
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
            # 注入隐身与真实请求头
            _apply_stealth(page)
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # 轻微人类行为：滚动与鼠标移动，避免立即输入触发检测
            try:
                page.wait_for_timeout(random.randint(300, 900))
                page.mouse.move(random.randint(50, 300), random.randint(50, 300))
                page.wait_for_timeout(random.randint(120, 420))
                page.evaluate("window.scrollBy(0, %d)" % random.randint(80, 300))
            except Exception:
                pass

            # 查找并聚焦输入框（支持主文档与 iframe）
            def _find_input_on_frame(fr):
                # 优先匹配含提示词的 textarea
                for sel in [
                    "textarea[placeholder*='发消息']",
                    "textarea[placeholder*='输入']",
                    "textarea[placeholder*='/ 选择技能']",
                ]:
                    loc = fr.locator(sel)
                    if loc.count() > 0:
                        return loc.first
                # 其次匹配常见及扩展的输入容器
                for sel in [
                    "textarea",
                    "[contenteditable='true']",
                    ".ProseMirror[contenteditable='true']",
                    "div[role='textbox']",
                    "div[data-placeholder]",
                    "div[placeholder]",
                    ".chat-input textarea",
                    ".chat-input [contenteditable='true']",
                ]:
                    l = fr.locator(sel)
                    if l.count() > 0:
                        return l.first
                return None

            input_loc = _find_input_on_frame(page)
            if not input_loc:
                # 跨 iframe 搜索
                for fr in page.frames:
                    try:
                        loc = _find_input_on_frame(fr)
                        if loc:
                            input_loc = loc
                            break
                    except Exception:
                        continue
            if not input_loc:
                # 尝试等待常见输入选择器一段时间（页面可能还在渲染）
                try:
                    candidates = ", ".join([
                        "textarea[placeholder*='发消息']",
                        "textarea[placeholder*='输入']",
                        "textarea[placeholder*='/ 选择技能']",
                        "textarea",
                        "[contenteditable='true']",
                        ".ProseMirror[contenteditable='true']",
                        "div[role='textbox']",
                        "div[data-placeholder]",
                        "div[placeholder]",
                        ".chat-input textarea",
                        ".chat-input [contenteditable='true']",
                    ])
                    page.wait_for_selector(candidates, timeout=15000)
                    loc = page.locator(candidates)
                    if loc.count() > 0:
                        input_loc = loc.first
                except Exception:
                    pass
            if not input_loc:
                print("未找到输入框，可能尚未登录或页面结构变化。")
                try:
                    ctx.close()
                except Exception:
                    pass
                return None

            # 人类化逐字输入助手（在函数作用域内，使用当前 page 与定位器）
            def _human_type(target_loc, text: str):
                # 为本次输入选择一个基础速度，并在每个字符上做小幅随机抖动
                base_ms = random.randint(20, 60)
                for ch in text:
                    delay_ms = max(10, base_ms + random.randint(-10, 10))
                    try:
                        # 优先使用定位器逐字输入，确保事件落在目标元素
                        target_loc.type(ch, delay=delay_ms)
                    except Exception:
                        # 退回到键盘输入
                        page.keyboard.type(ch, delay=delay_ms)
                    # 每个字符后更短的轻微停顿，加快总体时间
                    time.sleep(random.uniform(0.005, 0.03))
                    # 偶尔较长停顿，模拟思考但不影响效率
                    if random.random() < 0.02:
                        time.sleep(random.uniform(0.08, 0.25))

            # 点击并聚焦输入框，使用键盘输入（对 textarea 与 contenteditable 都通用）
            try:
                input_loc.scroll_into_view_if_needed(timeout=5000)
            except Exception:
                pass
            # 模拟类人点击与轻微滚动
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
            # 逐字输入并随机停顿，模拟人类打字
            _human_type(input_loc, prompt)

            # 使用键盘按 Enter 发送（确保针对当前焦点元素，兼容 iframe）
            sent = False
            try:
                time.sleep(random.uniform(0.15, 0.35))
                page.keyboard.press("Enter")
                sent = True
            except Exception:
                sent = False
            if not sent:
                # 回退尝试按钮点击
                for send_sel in [
                    "button:has-text('发送')",
                    "button[aria-label='发送']",
                    "button[type='submit']",
                ]:
                    try:
                        page.locator(send_sel).first.click()
                        sent = True
                        break
                    except Exception:
                        continue

            # 校验是否已发送：在页面中寻找用户消息文本
            def _user_message_appeared() -> bool:
                try:
                    # 直接按文本查找（可能有截断，使用包含匹配）
                    if page.get_by_text(prompt[:20]).count() > 0:
                        return True
                except Exception:
                    pass
                return False

            start_check = time.time()
            while time.time() - start_check < 5:
                if _user_message_appeared():
                    break
                time.sleep(0.5)

            if not _user_message_appeared():
                # 尝试其他快捷键发送
                for key in ["Control+Enter", "Meta+Enter", "Shift+Enter", "Enter"]:
                    try:
                        page.keyboard.press(key)
                        time.sleep(0.5)
                        if _user_message_appeared():
                            break
                    except Exception:
                        continue

            if not _user_message_appeared():
                # 尝试点击输入框附近的可能发送按钮
                try:
                    btn_near = input_loc.locator("xpath=../..//button").first
                    btn_near.click(timeout=3000)
                except Exception:
                    pass

            # 检测是否出现验证弹窗（常见关键字/第三方 iframe），如出现则提示人工处理
            try:
                possible_captcha = False
                captcha_selectors = [
                    "text=验证", "text=安全验证", "text=请完成验证",
                    "iframe[src*='geetest']", "iframe[src*='captcha']", "iframe[src*='hcaptcha']",
                ]
                for sel in captcha_selectors:
                    if page.locator(sel).count() > 0:
                        possible_captcha = True
                        break
                if possible_captcha and not headless:
                    print("检测到可能的验证弹窗，请在浏览器中完成验证后回车继续...")
                    try:
                        input("按回车继续...")
                    except Exception:
                        pass
            except Exception:
                pass

            # 等待并抓取回复
            response_selectors = [
                ".message-list .assistant .message-content",
                ".message-item .markdown-content",
                ".message-item .reply-content",
                ".message-container .content",
                ".markdown-body",
                "article",
            ]
            start_time = time.time()
            reply_text = None
            # 尝试基于用户消息定位：找到包含本次 prompt 的消息容器，取其下一个兄弟作为助手回复
            try:
                anchor = None
                anchor_loc = page.get_by_text(prompt[:20])
                if anchor_loc.count() > 0:
                    try:
                        anchor = anchor_loc.last
                    except Exception:
                        anchor = None
                if anchor:
                    # 找到用户消息所在的最邻近块级容器，再取其下一个兄弟节点
                    candidate = anchor.locator("xpath=ancestor::*[self::div or self::section or self::article][1]/following-sibling::*[self::div or self::section or self::article][1]")
                    if candidate.count() > 0:
                        try:
                            # 在候选容器中优先抓取 markdown/正文区域
                            inner = None
                            for sub in [".markdown", ".markdown-body", "article", "div:has(p)", "div:has(pre)"]:
                                loc2 = candidate.locator(sub)
                                if loc2.count() > 0:
                                    inner = loc2.first.inner_text()
                                    if inner and inner.strip():
                                        reply_text = inner.strip()
                                        break
                            if not reply_text:
                                txt = candidate.inner_text()
                                if txt and txt.strip():
                                    reply_text = txt.strip()
                        except Exception:
                            pass
            except Exception:
                pass
            # 优先使用完成工具条“编辑/分享”作为完成标志，并从其上方提取正文
            try:
                for complete_text in ["编辑", "分享"]:
                    loc = page.get_by_text(complete_text)
                    if loc.count() > 0:
                        try:
                            loc.last.wait_for(state="visible", timeout=wait_timeout_sec * 1000)
                            # 工具条出现后，取其上一个兄弟节点作为内容容器
                            content_node = loc.last.locator("xpath=preceding-sibling::*[1]")
                            if content_node.count() > 0:
                                text = content_node.inner_text()
                                if text and text.strip():
                                    reply_text = text.strip()
                                    break
                        except Exception:
                            continue
            except Exception:
                pass
            while time.time() - start_time < wait_timeout_sec:
                for sel in response_selectors:
                    loc = page.locator(sel)
                    cnt = loc.count()
                    if cnt > 0:
                        try:
                            text = loc.nth(cnt - 1).inner_text()
                            if text and text.strip():
                                reply_text = text.strip()
                                break
                        except Exception:
                            continue
                if reply_text:
                    break
                time.sleep(1)

            # 终极回退：从页面可见文本中基于用户消息截断提取回复
            if not reply_text:
                try:
                    body_text = page.locator("body").inner_text()
                    if body_text:
                        # 使用最后一次出现的用户消息作为切分点
                        idx = body_text.rfind(prompt[:20])
                        if idx != -1:
                            tail = body_text[idx + len(prompt[:20]):]
                            # 若出现完成工具条关键词，以其作为终止点
                            for end_kw in ["编辑", "分享"]:
                                end_idx = tail.find(end_kw)
                                if end_idx != -1:
                                    tail = tail[:end_idx]
                                    break
                            cleaned = tail.strip()
                            # 简单清理过多空白
                            cleaned = re.sub(r"[\s\u00A0]+", " ", cleaned)
                            if cleaned:
                                reply_text = cleaned
                except Exception:
                    pass

            ctx.close()
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
    elif args and args[0] == "chat":
        prompt = " ".join(args[1:]) or "请解释什么是人工智能"
        resp = chat_once_persistent(prompt, headless=False)
        if resp:
            print(f"回复: {resp}")
        else:
            print("未能获取到回复")
    else:
        print("用法:\n  python playwright_chat.py bootstrap\n  python playwright_chat.py chat \"你的问题\"\n")
    