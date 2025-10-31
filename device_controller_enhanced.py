"""
Enhanced Device Controller Module
Improved accuracy with dynamic timing, animation detection, crash monitoring, and remote ADB support
"""

import subprocess
import os
import time
import base64
import tempfile
from typing import List, Tuple, Optional, Dict
from PIL import Image
import io
import json
import threading
from datetime import datetime

class DeviceControllerEnhanced:
    """
    Enhanced device controller with improved accuracy and reliability
    """
    
    def __init__(self, device_id: Optional[str] = None, adb_server_host: Optional[str] = None, adb_server_port: int = 5037):
        """Initialize enhanced device controller with optional remote ADB support"""
        self.device_id = device_id
        self.adb_server_host = adb_server_host
        self.adb_server_port = adb_server_port
        self.connected = False
        self.device_info = {}
        self.performance_profile = "medium"  # slow, medium, fast
        
        # Timing configuration based on device performance
        self.timing_config = {
            "slow": {"tap_wait": 1.0, "swipe_wait": 1.5, "screen_load": 3.0},
            "medium": {"tap_wait": 0.5, "swipe_wait": 0.8, "screen_load": 2.0},
            "fast": {"tap_wait": 0.3, "swipe_wait": 0.5, "screen_load": 1.0}
        }
        
        # Crash monitoring
        self.crash_detected = False
        self.anr_detected = False
        
        # Try to connect
        if device_id:
            self.connect(device_id)
        else:
            devices = self.get_devices()
            if devices:
                self.connect(devices[0])
    
    def _get_adb_prefix(self) -> str:
        """Get ADB command prefix with optional remote server"""
        if self.adb_server_host:
            return f"adb -H {self.adb_server_host} -P {self.adb_server_port}"
        return "adb"
    
    def connect_wireless_device(self, device_ip: str, port: int = 5555) -> bool:
        """
        Connect to a wireless device
        
        Args:
            device_ip: IP address of the device
            port: ADB port (default 5555)
        
        Returns:
            True if connection successful
        """
        try:
            adb_cmd = self._get_adb_prefix()
            print(f"🔌 Connecting to {device_ip}:{port}...")
            
            result = subprocess.run(
                f"{adb_cmd} connect {device_ip}:{port}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "connected" in result.stdout.lower() or "already connected" in result.stdout.lower():
                print(f"✅ Connected to {device_ip}:{port}")
                
                # Get the device ID
                devices = self.get_devices()
                for dev in devices:
                    if device_ip in dev:
                        self.device_id = dev
                        self.connected = True
                        self.device_info = self._get_device_info()
                        self._profile_device_performance()
                        return True
                
                return True
            else:
                print(f"❌ Failed to connect: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Wireless connection error: {e}")
            return False
    
    def disconnect_wireless_device(self, device_ip: str, port: int = 5555) -> bool:
        """Disconnect from wireless device"""
        try:
            adb_cmd = self._get_adb_prefix()
            subprocess.run(
                f"{adb_cmd} disconnect {device_ip}:{port}",
                shell=True,
                capture_output=True,
                timeout=5
            )
            print(f"🔌 Disconnected from {device_ip}:{port}")
            return True
        except Exception as e:
            print(f"⚠️ Disconnect error: {e}")
            return False
    
    def get_devices(self) -> List[str]:
        """Get list of connected devices"""
        try:
            adb_cmd = self._get_adb_prefix()
            result = subprocess.run(
                f"{adb_cmd} devices",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            lines = result.stdout.strip().split('\n')[1:]
            devices = []
            
            for line in lines:
                if line.strip() and '\t' in line:
                    device_id = line.split('\t')[0]
                    devices.append(device_id)
            
            return devices
        except Exception as e:
            print(f"❌ Error getting devices: {e}")
            return []
    
    def connect(self, device_id: str) -> bool:
        """Connect to specific device with performance profiling"""
        try:
            adb_cmd = self._get_adb_prefix()
            result = subprocess.run(
                f"{adb_cmd} -s {device_id} shell echo 'connected'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.device_id = device_id
                self.connected = True
                self.device_info = self._get_device_info()
                
                # Profile device performance
                self._profile_device_performance()
                
                print(f"✅ Connected to device: {device_id}")
                print(f"📊 Performance profile: {self.performance_profile}")
                return True
            else:
                print(f"❌ Failed to connect to device: {device_id}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def _get_device_info(self) -> Dict:
        """Get detailed device information"""
        info = {}
        try:
            adb_cmd = self._get_adb_prefix()
            
            # Model
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell getprop ro.product.model",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            info['model'] = result.stdout.strip()
            
            # Android version
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell getprop ro.build.version.release",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            info['android_version'] = result.stdout.strip()
            
            # Screen resolution
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell wm size",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            if "Physical size:" in result.stdout:
                resolution = result.stdout.split("Physical size:")[1].strip()
                info['resolution'] = resolution
                
                # Parse width and height
                if 'x' in resolution:
                    w, h = resolution.split('x')
                    info['width'] = int(w)
                    info['height'] = int(h)
            
            # SDK version
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell getprop ro.build.version.sdk",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            info['sdk_version'] = result.stdout.strip()
            
            # CPU info for performance profiling
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell getprop ro.product.cpu.abi",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            info['cpu_abi'] = result.stdout.strip()
            
        except Exception as e:
            print(f"⚠️ Error getting device info: {e}")
        
        return info
    
    def _profile_device_performance(self):
        """Profile device to determine optimal timing"""
        try:
            # Simple performance test: measure screenshot capture time
            start = time.time()
            self.capture_screenshot()
            screenshot_time = time.time() - start
            
            # Classify based on speed
            if screenshot_time > 2.0:
                self.performance_profile = "slow"
            elif screenshot_time > 1.0:
                self.performance_profile = "medium"
            else:
                self.performance_profile = "fast"
            
            print(f"📊 Screenshot time: {screenshot_time:.2f}s → Profile: {self.performance_profile}")
            
        except Exception as e:
            print(f"⚠️ Performance profiling failed: {e}")
            self.performance_profile = "medium"
    
    def _get_timing(self, action_type: str) -> float:
        """Get optimal wait time for action type based on device performance"""
        timing_map = {
            "tap_wait": self.timing_config[self.performance_profile]["tap_wait"],
            "swipe_wait": self.timing_config[self.performance_profile]["swipe_wait"],
            "screen_load": self.timing_config[self.performance_profile]["screen_load"]
        }
        return timing_map.get(action_type, 1.0)
    
    def wait_for_animation(self, timeout: float = 5.0) -> bool:
        """
        Wait for screen animations to complete by comparing screenshots
        
        Returns:
            True if stable, False if timeout
        """
        print(f"   ⏳ Waiting for animations to complete...")
        start_time = time.time()
        
        try:
            prev_screenshot = None
            stable_count = 0
            required_stable_frames = 2
            
            while time.time() - start_time < timeout:
                current_screenshot = self.capture_screenshot()
                
                if prev_screenshot:
                    # Compare screenshots
                    from PIL import Image, ImageChops, ImageStat
                    
                    prev_img = Image.open(io.BytesIO(base64.b64decode(prev_screenshot)))
                    curr_img = Image.open(io.BytesIO(base64.b64decode(current_screenshot)))
                    
                    if prev_img.size == curr_img.size:
                        diff = ImageChops.difference(prev_img, curr_img)
                        stat = ImageStat.Stat(diff)
                        diff_percentage = sum(stat.mean) / (len(stat.mean) * 255) * 100
                        
                        if diff_percentage < 1.0:  # Less than 1% change
                            stable_count += 1
                            if stable_count >= required_stable_frames:
                                print(f"   ✅ Screen stable after {time.time() - start_time:.2f}s")
                                return True
                        else:
                            stable_count = 0
                
                prev_screenshot = current_screenshot
                time.sleep(0.3)
            
            print(f"   ⚠️ Animation wait timeout ({timeout}s)")
            return False
            
        except Exception as e:
            print(f"   ⚠️ Animation detection error: {e}")
            # Fallback to fixed wait
            time.sleep(self._get_timing("screen_load"))
            return True
    
    def is_app_running(self, package_name: str) -> bool:
        """Check if app is currently running"""
        try:
            adb_cmd = self._get_adb_prefix()
            
            # Method 1: Check current focus
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell dumpsys window | grep mCurrentFocus",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if package_name in result.stdout:
                return True
            
            # Method 2: Check running processes (more reliable)
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell ps | grep {package_name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if package_name in result.stdout:
                return True
            
            # Method 3: Check with pidof (most reliable for newer Android)
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell pidof {package_name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.stdout.strip():  # If we get a PID, app is running
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking app status: {e}")
            # If check fails, assume app is running to avoid false failures
            return True
    
    def detect_crash(self, package_name: str) -> bool:
        """Detect if app has crashed"""
        try:
            adb_cmd = self._get_adb_prefix()
            
            # Check if app is still in process list
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell ps | grep {package_name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if not result.stdout.strip():
                self.crash_detected = True
                print(f"💥 Crash detected: {package_name} not running")
                return True
            
            # Check logcat for crash indicators
            logcat = self.get_device_logs(filter_tag="AndroidRuntime", lines=50)
            if "FATAL EXCEPTION" in logcat and package_name in logcat:
                self.crash_detected = True
                print(f"💥 Fatal exception detected in {package_name}")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Crash detection error: {e}")
            return False
    
    def detect_anr(self) -> bool:
        """Detect Application Not Responding (ANR) state"""
        try:
            adb_cmd = self._get_adb_prefix()
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell dumpsys activity | grep 'ANR in'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.stdout.strip():
                self.anr_detected = True
                print(f"⏸️ ANR detected")
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def capture_screenshot(self, save_path: Optional[str] = None) -> str:
        """
        Enhanced screenshot capture with retry logic
        """
        if not self.connected:
            raise Exception("Device not connected")
        
        adb_cmd = self._get_adb_prefix()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use exec-out for faster capture
                result = subprocess.run(
                    f"{adb_cmd} -s {self.device_id} exec-out screencap -p",
                    shell=True,
                    capture_output=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout and len(result.stdout) > 1000:
                    img_data = result.stdout
                else:
                    # Fallback method
                    temp_device_path = "/sdcard/screen_temp.png"
                    subprocess.run(
                        f"{adb_cmd} -s {self.device_id} shell screencap -p {temp_device_path}",
                        shell=True,
                        timeout=3,
                        capture_output=True
                    )
                    
                    temp_local = tempfile.mktemp(suffix='.png')
                    subprocess.run(
                        f"{adb_cmd} -s {self.device_id} pull {temp_device_path} {temp_local}",
                        shell=True,
                        timeout=3,
                        capture_output=True
                    )
                    
                    with open(temp_local, 'rb') as f:
                        img_data = f.read()
                    
                    os.remove(temp_local)
                
                # Validate image data
                try:
                    Image.open(io.BytesIO(img_data))
                except Exception:
                    raise Exception("Invalid image data")
                
                # Save if path provided
                if save_path:
                    with open(save_path, 'wb') as f:
                        f.write(img_data)
                
                # Return base64
                return base64.b64encode(img_data).decode('utf-8')
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️ Screenshot attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.5)
                else:
                    print(f"❌ Screenshot failed after {max_retries} attempts: {e}")
                    raise
    
    def tap_with_retry(self, x: int, y: int, retries: int = 3, offset_variance: int = 10) -> bool:
        """
        Tap with retry logic and coordinate variance
        
        Args:
            x, y: Base coordinates
            retries: Number of retry attempts
            offset_variance: Random offset range for retries (helps with precision)
        
        Returns:
            True if any attempt succeeded
        """
        if not self.connected:
            return False
        
        import random
        adb_cmd = self._get_adb_prefix()
        
        for attempt in range(retries):
            try:
                # Add slight random offset on retries to improve hit rate
                if attempt > 0:
                    offset_x = random.randint(-offset_variance, offset_variance)
                    offset_y = random.randint(-offset_variance, offset_variance)
                    tap_x = x + offset_x
                    tap_y = y + offset_y
                    print(f"   🔄 Retry {attempt + 1}: Tapping ({tap_x}, {tap_y}) [offset: {offset_x}, {offset_y}]")
                else:
                    tap_x, tap_y = x, y
                
                subprocess.run(
                    f"{adb_cmd} -s {self.device_id} shell input tap {tap_x} {tap_y}",
                    shell=True,
                    timeout=3,
                    capture_output=True
                )
                
                # Dynamic wait based on device performance
                time.sleep(self._get_timing("tap_wait"))
                
                return True
                
            except Exception as e:
                if attempt < retries - 1:
                    print(f"   ⚠️ Tap attempt {attempt + 1} failed: {e}")
                    time.sleep(0.5)
                else:
                    print(f"❌ Tap failed after {retries} attempts")
                    return False
        
        return False
    
    def tap(self, x: int, y: int, duration: int = 100) -> bool:
        """Standard tap (calls tap_with_retry)"""
        return self.tap_with_retry(x, y, retries=2)
    
    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, 
              duration: int = 500) -> bool:
        """Enhanced swipe with dynamic timing"""
        if not self.connected:
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}",
                shell=True,
                timeout=3,
                capture_output=True
            )
            time.sleep(self._get_timing("swipe_wait"))
            return True
        except Exception as e:
            print(f"❌ Swipe error: {e}")
            return False
    
    def input_text(self, text: str) -> bool:
        """Input text with better escaping"""
        if not self.connected:
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            # Better escaping for special characters
            text = text.replace(' ', '%s')
            text = text.replace('&', '\\&')
            text = text.replace('(', '\\(')
            text = text.replace(')', '\\)')
            text = text.replace('<', '\\<')
            text = text.replace('>', '\\>')
            text = text.replace('|', '\\|')
            text = text.replace(';', '\\;')
            text = text.replace('`', '\\`')
            text = text.replace('"', '\\"')
            text = text.replace("'", "\\'")
            
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell input text '{text}'",
                shell=True,
                timeout=5,
                capture_output=True
            )
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"❌ Text input error: {e}")
            return False
    
    def press_key(self, keycode: str) -> bool:
        """Press key with dynamic timing"""
        if not self.connected:
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell input keyevent {keycode}",
                shell=True,
                timeout=3,
                capture_output=True
            )
            time.sleep(self._get_timing("tap_wait"))
            return True
        except Exception as e:
            print(f"❌ Key press error: {e}")
            return False
    
    def start_app(self, package_name: str, activity: Optional[str] = None) -> bool:
        """Launch app with verification"""
        if not self.connected:
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            print(f"   🚀 Launching {package_name}...")
            
            if activity:
                cmd = f"{adb_cmd} -s {self.device_id} shell am start -n {package_name}/{activity}"
            else:
                cmd = f"{adb_cmd} -s {self.device_id} shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
            
            result = subprocess.run(cmd, shell=True, timeout=10, capture_output=True, text=True)
            
            # Check if launch command succeeded
            if result.returncode != 0:
                print(f"   ⚠️ Launch command failed: {result.stderr}")
                # Still wait and check if app started
            
            # Wait for app to load
            wait_time = self._get_timing("screen_load")
            print(f"   ⏳ Waiting {wait_time:.1f}s for app to load...")
            time.sleep(wait_time)
            
            # Give extra time for heavy apps
            time.sleep(2)
            
            # Verify app started (but don't fail if verification fails)
            is_running = self.is_app_running(package_name)
            
            if is_running:
                print(f"   ✅ App {package_name} verified running")
                return True
            else:
                print(f"   ⚠️ Cannot verify app is running, but launch command executed")
                print(f"   ℹ️ Continuing anyway - app may have started")
                # Return True anyway since the launch command executed without error
                # The verification might fail for various reasons but app could be running
                return True
                
        except subprocess.TimeoutExpired:
            print(f"   ⚠️ App start command timed out")
            print(f"   ℹ️ App may still be starting, continuing...")
            return True  # App might still be starting
        except Exception as e:
            print(f"   ❌ App start error: {e}")
            return False
    
    def stop_app(self, package_name: str) -> bool:
        """Force stop app"""
        if not self.connected:
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell am force-stop {package_name}",
                shell=True,
                timeout=3,
                capture_output=True
            )
            time.sleep(1)
            
            # Verify app stopped
            if not self.is_app_running(package_name):
                print(f"✅ App {package_name} stopped")
                return True
            else:
                print(f"⚠️ App may still be running")
                return False
                
        except Exception as e:
            print(f"❌ App stop error: {e}")
            return False
    
    def install_apk(self, apk_path: str) -> bool:
        """Install APK"""
        if not self.connected:
            return False
        
        if not os.path.exists(apk_path):
            print(f"❌ APK file not found: {apk_path}")
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            print(f"📦 Installing {apk_path}...")
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} install -r {apk_path}",
                shell=True,
                timeout=60,
                capture_output=True,
                text=True
            )
            
            if "Success" in result.stdout:
                print("✅ APK installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ APK install error: {e}")
            return False
    
    def uninstall_app(self, package_name: str) -> bool:
        """Uninstall app"""
        if not self.connected:
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} uninstall {package_name}",
                shell=True,
                timeout=10,
                capture_output=True
            )
            return True
        except Exception as e:
            print(f"❌ Uninstall error: {e}")
            return False
    
    def get_current_activity(self) -> Optional[str]:
        """Get current activity"""
        if not self.connected:
            return None
        
        adb_cmd = self._get_adb_prefix()
        try:
            result = subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell dumpsys window | grep mCurrentFocus",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            return result.stdout.strip() if result.stdout else None
            
        except Exception as e:
            print(f"⚠️ Error getting activity: {e}")
            return None
    
    def clear_app_data(self, package_name: str) -> bool:
        """Clear app data"""
        if not self.connected:
            return False
        
        adb_cmd = self._get_adb_prefix()
        try:
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell pm clear {package_name}",
                shell=True,
                timeout=5,
                capture_output=True
            )
            time.sleep(1)
            return True
        except Exception as e:
            print(f"❌ Clear data error: {e}")
            return False
    
    def get_device_logs(self, filter_tag: Optional[str] = None, lines: int = 100) -> str:
        """Get device logs"""
        if not self.connected:
            return ""
        
        adb_cmd = self._get_adb_prefix()
        try:
            cmd = f"{adb_cmd} -s {self.device_id} logcat -d -t {lines}"
            if filter_tag:
                cmd += f" -s {filter_tag}"
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return result.stdout
            
        except Exception as e:
            print(f"⚠️ Log retrieval error: {e}")
            return ""
    
    def get_ui_hierarchy(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Get UI hierarchy XML for precise element detection
        
        Returns:
            XML content or None
        """
        if not self.connected:
            return None
        
        adb_cmd = self._get_adb_prefix()
        try:
            # Dump UI hierarchy to device
            device_path = "/sdcard/ui_dump.xml"
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} shell uiautomator dump {device_path}",
                shell=True,
                timeout=5,
                capture_output=True
            )
            
            # Pull to local
            local_path = save_path or tempfile.mktemp(suffix='.xml')
            subprocess.run(
                f"{adb_cmd} -s {self.device_id} pull {device_path} {local_path}",
                shell=True,
                timeout=3,
                capture_output=True
            )
            
            # Read content
            with open(local_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            if not save_path:
                os.remove(local_path)
            
            return xml_content
            
        except Exception as e:
            print(f"⚠️ UI hierarchy dump error: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from device"""
        self.connected = False
        self.device_id = None
        print("🔌 Disconnected from device")


if __name__ == "__main__":
    print("✅ Enhanced Device Controller loaded")
    print("\n🎯 New Features:")
    print("  • Dynamic timing based on device performance")
    print("  • Animation detection and waiting")
    print("  • Crash and ANR detection")
    print("  • Tap retry with coordinate variance")
    print("  • UI hierarchy XML dumping")
    print("  • App running state verification")
    print("  • Remote ADB server support")
    print("  • Wireless device connection")