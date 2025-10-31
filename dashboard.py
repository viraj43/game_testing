"""
AI Game Testing Platform - Web Dashboard
Main user interface for AI-powered mobile game testing
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime
import time

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from ai_engine_enhanced import AIGameTestEngineEnhanced
from device_controller_enhanced import DeviceControllerEnhanced  
from game_test_executor_enhanced import GameTestExecutorEnhanced

# Alias to maintain compatibility
AIGameTestEngine = AIGameTestEngineEnhanced
DeviceController = DeviceControllerEnhanced
GameTestExecutor = GameTestExecutorEnhanced

# Page config
st.set_page_config(
    page_title="AI Game Testing Platform",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'test_running' not in st.session_state:
    st.session_state.test_running = False
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'device_connected' not in st.session_state:
    st.session_state.device_connected = False
if 'device_info' not in st.session_state:
    st.session_state.device_info = {}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .step-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .ai-provider-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px;
    }
    .connection-guide {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Game Testing Platform</h1>
    <p style="font-size: 18px; margin-top: 10px;">Test your mobile games with artificial intelligence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Device Connection
with st.sidebar:
    st.header("🔌 Device Connection")
    
    # Connection method tabs
    connection_tab1, connection_tab2 = st.tabs(["USB", "WiFi"])
    
    with connection_tab1:
        st.markdown("**USB Connection**")
        if st.button("🔍 Scan for USB Devices", use_container_width=True):
            with st.spinner("Scanning..."):
                try:
                    devices = DeviceController().get_devices()
                    if devices:
                        st.session_state.available_devices = devices
                        st.success(f"Found {len(devices)} device(s)")
                    else:
                        st.warning("No devices found. Please connect a device via USB.")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Device selection
        if 'available_devices' in st.session_state and st.session_state.available_devices:
            selected_device = st.selectbox(
                "Select Device",
                st.session_state.available_devices,
                key="usb_device_select"
            )
            
            if st.button("Connect to Device", use_container_width=True, key="usb_connect"):
                with st.spinner("Connecting..."):
                    try:
                        controller = DeviceController(selected_device)
                        if controller.connected:
                            st.session_state.device_connected = True
                            st.session_state.device_info = controller.device_info
                            st.session_state.selected_device = selected_device
                            st.success("✅ Connected!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
    
    with connection_tab2:
        st.markdown("**Wireless Connection**")
        
        with st.expander("📖 Setup Instructions", expanded=False):
            st.markdown("""
            <div class="connection-guide">
            <b>How to connect wirelessly:</b><br><br>
            
            <b>Step 1:</b> Connect your phone via USB to <i>any computer</i><br>
            <b>Step 2:</b> On that computer, run:<br>
            <code>adb tcpip 5555</code><br><br>
            
            <b>Step 3:</b> Find your phone's IP address:<br>
            • Settings → About Phone → Status → IP Address<br>
            • Or run: <code>adb shell ip addr show wlan0</code><br><br>
            
            <b>Step 4:</b> Enter IP below and click Connect<br><br>
            
            ⚠️ <b>Important:</b> Your phone and this server must be on the <b>same WiFi network</b>!
            </div>
            """, unsafe_allow_html=True)
        
        device_ip = st.text_input(
            "Device IP Address",
            placeholder="192.168.1.100",
            help="Enter your Android device's WiFi IP address",
            key="wifi_ip"
        )
        
        device_port = st.number_input(
            "ADB Port",
            value=5555,
            min_value=5555,
            max_value=5585,
            help="Default ADB wireless port is 5555",
            key="wifi_port"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔌 Connect", use_container_width=True, key="wifi_connect"):
                if device_ip:
                    with st.spinner(f"Connecting to {device_ip}:{device_port}..."):
                        try:
                            controller = DeviceControllerEnhanced()
                            if controller.connect_wireless_device(device_ip, device_port):
                                # Refresh device list
                                time.sleep(1)
                                devices = controller.get_devices()
                                if devices:
                                    st.session_state.available_devices = devices
                                    # Auto-select the wireless device
                                    for dev in devices:
                                        if device_ip in dev:
                                            st.session_state.selected_device = dev
                                            st.session_state.device_connected = True
                                            st.session_state.device_info = controller.device_info
                                            st.success(f"✅ Connected to {device_ip}!")
                                            st.rerun()
                                            break
                                else:
                                    st.warning("Device connected but not showing in list. Try USB scan.")
                            else:
                                st.error("❌ Connection failed. Check:\n- IP address is correct\n- Device and server on same network\n- ADB over TCP enabled (adb tcpip 5555)")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please enter device IP address")
        
        with col2:
            if device_ip and st.button("Disconnect", use_container_width=True, key="wifi_disconnect"):
                try:
                    controller = DeviceControllerEnhanced()
                    controller.disconnect_wireless_device(device_ip, device_port)
                    if st.session_state.device_connected:
                        st.session_state.device_connected = False
                        st.session_state.device_info = {}
                        st.success(f"Disconnected from {device_ip}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Show device info if connected
    if st.session_state.device_connected:
        st.success("📱 Device Connected")
        
        # Determine connection type
        device_id = st.session_state.get('selected_device', '')
        connection_type = "WiFi" if ':' in device_id else "USB"
        
        st.info(f"**Connection:** {connection_type}\n\n"
                f"**Device:** {device_id}\n\n"
                f"**Model:** {st.session_state.device_info.get('model', 'Unknown')}\n\n"
                f"**Android:** {st.session_state.device_info.get('android_version', 'Unknown')}\n\n"
                f"**Resolution:** {st.session_state.device_info.get('resolution', 'Unknown')}")
        
        if st.button("🔌 Disconnect Device", use_container_width=True):
            st.session_state.device_connected = False
            st.session_state.device_info = {}
            st.session_state.selected_device = None
            st.rerun()
    
    st.markdown("---")
    
    # AI Configuration
    st.header("🧠 AI Configuration")
    
    ai_provider = st.selectbox(
        "AI Provider",
        ["Claude (Anthropic)", "GPT-4 (OpenAI)", "Gemini (Google)"],
        help="Choose which AI to use for test execution"
    )
    
    # Map display names to internal names
    provider_map = {
        "Claude (Anthropic)": "claude",
        "GPT-4 (OpenAI)": "openai",
        "Gemini (Google)": "gemini"
    }
    
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your API key for the selected provider"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        st.session_state.ai_provider = provider_map[ai_provider]
    
    # Display AI provider info
    if 'ai_provider' in st.session_state:
        provider_info = {
            "claude": ("Claude Sonnet 4.5", "#764ba2", "claude-sonnet-4-5-20250929"),
            "openai": ("gpt-4o", "#10a37f", "gpt-4o"),
            "gemini": ("gemini-flash-latest", "#4285f4", "gemini-flash-latest")
        }
        
        if st.session_state.ai_provider in provider_info:
            name, color, model = provider_info[st.session_state.ai_provider]
            st.markdown(f"""
            <div style='background: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-top: 10px;'>
                <b>Using: {name}</b><br>
                <small>{model}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Help
    with st.expander("ℹ️ Quick Help"):
        st.markdown("""
        **How to use:**
        1. Connect your Android device (USB or WiFi)
        2. Select AI provider
        3. Enter your AI API key
        4. Enter game package name
        5. Write test in natural language
        6. Click "Run AI Test"
        
        **Example instructions:**
```
        Open the game
        Tap "Play" button
        Complete level 1
        Verify coins increased
```
        
        **Supported AI Features:**
        - 🎯 Element detection
        - 📸 Screenshot analysis
        - ✅ Action verification
        - 🐛 Automatic debugging
        - 📊 Test strategy generation
        """)

# Main content area
if not st.session_state.device_connected:
    st.warning("⚠️ Please connect a device to get started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🎮 Any Game</h3>
            <p>Test casual games, action games, puzzle games - anything!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>✍️ Natural Language</h3>
            <p>Just describe what to test in plain English</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 AI Powered</h3>
            <p>AI sees, understands, and tests your game intelligently</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show AI capabilities
    st.subheader("🧠 AI-Powered Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Visual Understanding:**
        - 🔍 Automatically finds UI elements
        - 📍 Calculates tap coordinates
        - 🎯 Identifies game objects
        - 📊 Analyzes game state
        """)
    
    with col2:
        st.markdown("""
        **Intelligent Testing:**
        - ✅ Verifies actions succeeded
        - 🐛 Debugs issues automatically
        - 📈 Generates test strategies
        - 📝 Creates detailed reports
        """)
    
    st.markdown("---")
    
    # Connection help section
    st.subheader("📱 Connection Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>🔌 USB Connection (Recommended)</h4>
            <p><b>Pros:</b> Fast, reliable, no setup needed<br>
            <b>Cons:</b> Requires physical connection</p>
            <p><b>Steps:</b><br>
            1. Connect device via USB cable<br>
            2. Enable USB debugging<br>
            3. Click "Scan for USB Devices"</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>📡 WiFi Connection</h4>
            <p><b>Pros:</b> Wireless, convenient<br>
            <b>Cons:</b> Requires initial USB setup</p>
            <p><b>Steps:</b><br>
            1. Connect via USB first<br>
            2. Run: <code>adb tcpip 5555</code><br>
            3. Find device IP address<br>
            4. Use WiFi tab in sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("💡 Connect a device from the sidebar to start testing!")

else:
    # Main testing interface
    st.header("🎮 Game Test Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        game_package = st.text_input(
            "Game Package Name",
            placeholder="com.example.busfrenzy",
            help="The package name of your game (e.g., com.yourcompany.yourgame)"
        )
    
    with col2:
        apk_file = st.file_uploader(
            "Upload APK (Optional)",
            type=['apk'],
            help="Upload APK if you want to install/update the game first"
        )
    
    st.markdown("---")
    
    # Test Strategy Generator
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("✍️ Test Instructions")
        st.caption("Describe what you want to test in natural language. The AI will understand and execute!")
    
    with col2:
        if game_package and st.session_state.get('api_key'):
            if st.button("🧠 Generate Test Strategy", use_container_width=True):
                with st.spinner("AI is generating test strategy..."):
                    try:
                        ai_engine = AIGameTestEngine(
                            ai_provider=st.session_state.ai_provider,
                            api_key=st.session_state.api_key
                        )
                        
                        test_objective = st.text_input(
                            "What do you want to test?",
                            placeholder="e.g., Basic gameplay flow"
                        )
                        
                        if test_objective:
                            strategy = ai_engine.generate_test_strategy(
                                game_name=game_package.split('.')[-1],
                                test_objective=test_objective
                            )
                            st.session_state.generated_strategy = strategy
                            st.success("✅ Strategy generated!")
                    except Exception as e:
                        st.error(f"Error generating strategy: {e}")
    
    # Tab for different input methods
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Write Instructions", "📋 Use Template", "🧠 AI Generated", "💡 Examples"])
    
    with tab1:
        test_instructions = st.text_area(
            "Test Instructions",
            height=300,
            placeholder="""Example:

Open Bus Frenzy game
Skip tutorial if present
Tap "Play" button
Start Level 1
Pick up all passengers (should be 5)
Drive to the destination station
Complete the level
Verify that I earned 100 coins
Check that Level 2 is now unlocked
Play 2 more levels
Report any bugs or issues found
""",
            label_visibility="collapsed"
        )
    
    with tab2:
        st.markdown("**Select a template:**")
        
        templates = {
            "Basic Gameplay Test": """Open the game
Skip tutorial
Play first level
Complete all objectives
Verify level completion screen
Check rewards were given""",
            
            "Login Flow Test": """Open the game
Find and tap login button
Enter username: testuser@email.com
Enter password: Test123
Tap login button
Verify successful login
Check user profile is visible""",
            
            "IAP Test": """Open the game
Navigate to shop
Find coin package (100 coins)
Tap to view details
Verify price is displayed
Cancel purchase
Go back to main menu""",
            
            "Multi-Level Test": """Open the game
Play levels 1 through 5
For each level:
  - Complete all objectives
  - Collect all items
  - Verify level completion
Report final score and coins""",
        }
        
        selected_template = st.selectbox("Choose template", list(templates.keys()))
        
        if st.button("Use This Template"):
            st.session_state.template_text = templates[selected_template]
            st.rerun()
        
        if 'template_text' in st.session_state:
            test_instructions = st.text_area(
                "Edit template:",
                value=st.session_state.template_text,
                height=300
            )
    
    with tab3:
        if 'generated_strategy' in st.session_state:
            test_instructions = st.text_area(
                "AI Generated Strategy:",
                value=st.session_state.generated_strategy,
                height=300
            )
        else:
            st.info("Click 'Generate Test Strategy' button above to let AI create a test plan for you!")
            test_instructions = ""
    
    with tab4:
        st.markdown("""
        **Example Test Instructions:**
        
        **For Bus Frenzy:**
```
        Open Bus Frenzy
        Tap Play button
        Start Level 1
        Pick up 5 passengers
        Drive to destination
        Complete level in under 2 minutes
        Verify got 100 coins
```
        
        **For Match-3 Game:**
```
        Open Candy Crush style game
        Play level 1
        Make 5 matches
        Use power-up if available
        Complete level
        Check star rating
```
        
        **For Action Game:**
```
        Launch game
        Skip intro
        Start first mission
        Collect 10 coins
        Defeat 3 enemies
        Complete mission
        Verify rewards
```
        """)
        test_instructions = ""
    
    # Preview parsed steps
    if test_instructions and st.session_state.get('api_key'):
        with st.expander("🔍 Preview AI-Parsed Test Steps"):
            if st.button("Parse Instructions"):
                with st.spinner("AI is analyzing instructions..."):
                    try:
                        ai_engine = AIGameTestEngine(
                            ai_provider=st.session_state.ai_provider,
                            api_key=st.session_state.api_key
                        )
                        
                        parsed_steps = ai_engine.parse_test_instructions(test_instructions)
                        
                        st.success(f"✅ Parsed into {len(parsed_steps)} steps")
                        
                        for step in parsed_steps:
                            st.markdown(f"""
                            <div class="step-box">
                                <b>Step {step['step_number']}:</b> {step['description']}<br>
                                <small>Action: {step['action']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error parsing: {e}")
    
    st.markdown("---")
    
    # Run test button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run_button = st.button(
            "🚀 Run AI Test",
            type="primary",
            use_container_width=True,
            # disabled=st.session_state.test_running or not game_package or not test_instructions
        )
    
    with col2:
        if st.session_state.test_running:
            if st.button("⏹️ Stop Test", use_container_width=True):
                st.session_state.test_running = False
                st.rerun()
    
    with col3:
        status_color = "🔴" if st.session_state.test_running else "🟢"
        status_text = "Running" if st.session_state.test_running else "Ready"
        st.caption(f"{status_color} {status_text}")
    
    # Execute test
    if run_button:
        if not st.session_state.get('api_key'):
            st.error("⚠️ Please enter your AI API key in the sidebar first!")
        else:
            st.session_state.test_running = True
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                st.info(f"🤖 AI ({st.session_state.ai_provider.upper()}) is analyzing your test instructions...")
                
                try:
                    # Save APK if uploaded
                    apk_path = None
                    if apk_file:
                        apk_path = f"temp_{apk_file.name}"
                        with open(apk_path, "wb") as f:
                            f.write(apk_file.getbuffer())
                    
                    # Initialize executor with AI engine
                    executor = GameTestExecutor(
                        ai_provider=st.session_state.ai_provider,
                        api_key=st.session_state.api_key,
                        device_id=st.session_state.selected_device
                    )
                    
                    # Run test with progress updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Initializing AI test engine...")
                    progress_bar.progress(10)
                    
                    # Execute test
                    test_results = executor.execute_test(
                        game_package=game_package,
                        test_instructions=test_instructions,
                        apk_path=apk_path
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Test completed!")
                    
                    # Store results
                    st.session_state.test_results = test_results
                    st.session_state.test_running = False
                    
                    # Generate report
                    report_path = executor.generate_report(test_results)
                    st.session_state.report_path = report_path
                    
                    # Clean up temp APK
                    if apk_path and os.path.exists(apk_path):
                        os.remove(apk_path)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
                    st.exception(e)
                    st.session_state.test_running = False
    
    # Display results
    if st.session_state.test_results:
        st.markdown("---")
        st.header("📊 Test Results")
        
        results = st.session_state.test_results
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = results.get('status', 'unknown')
            status_emoji = "✅" if status == "passed" else "❌"
            st.metric("Status", f"{status_emoji} {status.upper()}")
        
        with col2:
            success_rate = results.get('success_rate', 0)
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            duration = results.get('duration_seconds', 0)
            st.metric("Duration", f"{duration:.1f}s")
        
        with col4:
            steps = results.get('passed_steps', 0)
            total = results.get('total_steps', 0)
            st.metric("Steps", f"{steps}/{total}")
        
        # Detailed steps
        st.subheader("Test Steps")
        
        for i, step_result in enumerate(results.get('steps_executed', []), 1):
            success = step_result.get('success', False)
            step_info = step_result.get('step', {})
            
            with st.expander(
                f"{'✅' if success else '❌'} Step {i}: {step_info.get('description', 'Unknown')}",
                expanded=not success
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Action:** {step_info.get('action', 'N/A')}")
                    st.write(f"**Target:** {step_info.get('target', 'N/A')}")
                    st.write(f"**Attempts:** {step_result.get('attempts', 0)}")
                    st.write(f"**Duration:** {step_result.get('duration', 0):.2f}s")
                    
                    if not success:
                        st.error(f"**Error:** {step_result.get('error', 'Unknown error')}")
                    
                    # AI observations
                    observations = step_result.get('ai_observations', [])
                    if observations:
                        st.write("**🧠 AI Analysis:**")
                        for obs in observations:
                            st.json(obs)
                    
                    # Element coordinates if found
                    if 'element_coordinates' in step_result:
                        coords = step_result['element_coordinates']
                        st.write(f"**📍 Element Location:** ({coords[0]}, {coords[1]})")
                
                with col2:
                    # Screenshots
                    screenshots = step_result.get('screenshots', [])
                    for ss in screenshots[:2]:
                        if os.path.exists(ss):
                            st.image(ss, caption=os.path.basename(ss), use_column_width=True)
        
        # AI Debug Analysis for failures
        if results.get('status') == 'failed':
            st.markdown("---")
            st.subheader("🐛 AI Debug Analysis")
            
            if 'debug_analysis' in results:
                debug = results['debug_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Diagnosis:**")
                    st.write(debug.get('diagnosis', 'N/A'))
                    
                    st.markdown("**Root Cause:**")
                    st.write(debug.get('root_cause', 'N/A'))
                
                with col2:
                    st.markdown("**Recovery Steps:**")
                    for step in debug.get('recovery_steps', []):
                        st.write(f"- {step}")
                    
                    st.markdown("**Prevention:**")
                    st.write(debug.get('prevention', 'N/A'))
        
        # Download report
        st.markdown("---")
        
        if 'report_path' in st.session_state and os.path.exists(st.session_state.report_path):
            with open(st.session_state.report_path, 'r', encoding='utf-8') as f:
                report_html = f.read()
            
            st.download_button(
                label="📥 Download Full HTML Report",
                data=report_html,
                file_name=f"test_report_{results.get('test_id', 'unknown')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        # Run another test
        if st.button("🔄 Run Another Test", use_container_width=True):
            st.session_state.test_results = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 AI Game Testing Platform | Powered by Claude, GPT-4 & Gemini</p>
    <p style='font-size: 12px;'>Test your mobile games with artificial intelligence</p>
</div>
""", unsafe_allow_html=True)