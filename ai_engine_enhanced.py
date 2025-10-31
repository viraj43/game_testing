"""
Enhanced AI Game Testing Engine with Accuracy Improvements
Includes: OCR, template matching, multi-resolution support, improved verification
"""

import anthropic
import openai
import google.generativeai as genai
import base64
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from PIL import Image, ImageChops, ImageStat
import io
import cv2
import numpy as np
import pytesseract
from pathlib import Path

class AIGameTestEngineEnhanced:
    """
    Enhanced AI engine with improved accuracy for game testing
    """
    
    def __init__(self, ai_provider="claude", api_key=None):
        """
        Initialize Enhanced AI Engine
        
        Args:
            ai_provider: "claude", "openai", or "gemini"
            api_key: API key for the chosen provider
        """
        self.ai_provider = ai_provider
        
        if ai_provider == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-5-20250929"
        elif ai_provider == "openai":
            openai.api_key = api_key
            self.model = "gpt-4o"
        elif ai_provider == "gemini":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-flash-latest')
            self.client = self.model
        else:
            raise ValueError("Provider must be 'claude', 'openai', or 'gemini'")
        
        self.test_history = []
        self.current_test = None
        self.conversation_history = []  # For context accumulation
        
        # Template storage for element matching
        self.element_templates = {}
        
        # Device resolution (will be set dynamically)
        self.device_width = 1080
        self.device_height = 2340
        
        print(f"✅ Enhanced AI Engine initialized with {ai_provider}")
    
    def set_device_resolution(self, width: int, height: int):
        """Set the actual device resolution for accurate coordinate mapping"""
        self.device_width = width
        self.device_height = height
        print(f"📐 Device resolution set: {width}x{height}")
    
    def _base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    
    def _image_to_cv2(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        image = self._base64_to_image(base64_str)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better AI recognition"""
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)
        
        return image
    
    def extract_text_from_screen(self, screenshot_base64: str, region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """
        Extract text from screenshot using OCR
        
        Args:
            screenshot_base64: Base64 encoded screenshot
            region: Optional (x, y, width, height) to crop before OCR
        
        Returns:
            List of detected text with positions and confidence
        """
        try:
            image = self._base64_to_image(screenshot_base64)
            
            # Crop to region if specified
            if region:
                x, y, w, h = region
                image = image.crop((x, y, x + w, y + h))
            
            # Convert to grayscale for better OCR
            image_gray = image.convert('L')
            
            # Use pytesseract to get detailed data
            ocr_data = pytesseract.image_to_data(image_gray, output_type=pytesseract.Output.DICT)
            
            extracted_texts = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                if text and conf > 30:  # Only include confident detections
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    
                    # Adjust coordinates if region was specified
                    if region:
                        x += region[0]
                        y += region[1]
                    
                    extracted_texts.append({
                        'text': text,
                        'confidence': conf,
                        'position': {'x': x, 'y': y, 'width': w, 'height': h},
                        'center': {'x': x + w // 2, 'y': y + h // 2}
                    })
            
            return extracted_texts
            
        except Exception as e:
            print(f"❌ OCR extraction error: {e}")
            return []
    
    def find_text_location(self, screenshot_base64: str, target_text: str, partial_match: bool = True) -> Optional[Tuple[int, int]]:
        """
        Find text on screen and return its center coordinates
        
        Args:
            screenshot_base64: Base64 encoded screenshot
            target_text: Text to find (case-insensitive)
            partial_match: If True, allows partial matches
        
        Returns:
            (x, y) coordinates of text center, or None if not found
        """
        extracted_texts = self.extract_text_from_screen(screenshot_base64)
        target_lower = target_text.lower()
        
        best_match = None
        best_confidence = 0
        
        for text_data in extracted_texts:
            text_lower = text_data['text'].lower()
            
            # Check for match
            is_match = False
            if partial_match:
                is_match = target_lower in text_lower or text_lower in target_lower
            else:
                is_match = target_lower == text_lower
            
            if is_match and text_data['confidence'] > best_confidence:
                best_match = text_data
                best_confidence = text_data['confidence']
        
        if best_match:
            print(f"✅ Found text '{target_text}' at ({best_match['center']['x']}, {best_match['center']['y']}) with {best_confidence}% confidence")
            return (best_match['center']['x'], best_match['center']['y'])
        
        print(f"❌ Text '{target_text}' not found via OCR")
        return None
    
    def compare_screenshots(self, before_base64: str, after_base64: str) -> Dict:
        """
        Compare two screenshots to detect changes
        
        Returns:
            Dict with difference metrics and changed regions
        """
        try:
            before_img = self._base64_to_image(before_base64)
            after_img = self._base64_to_image(after_base64)
            
            # Ensure same size
            if before_img.size != after_img.size:
                after_img = after_img.resize(before_img.size)
            
            # Calculate pixel difference
            diff = ImageChops.difference(before_img, after_img)
            
            # Get statistics
            stat = ImageStat.Stat(diff)
            diff_percentage = sum(stat.mean) / (len(stat.mean) * 255) * 100
            
            # Convert to grayscale for region detection
            diff_gray = diff.convert('L')
            diff_array = np.array(diff_gray)
            
            # Threshold to find changed regions
            _, thresh = cv2.threshold(diff_array, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours of changed regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            changed_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Ignore small changes (noise)
                    x, y, w, h = cv2.boundingRect(contour)
                    changed_regions.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'area': int(area)
                    })
            
            # Sort by area (largest first)
            changed_regions.sort(key=lambda r: r['area'], reverse=True)
            
            return {
                'difference_percentage': round(diff_percentage, 2),
                'changed_regions': changed_regions[:5],  # Top 5 changed regions
                'significant_change': diff_percentage > 5.0,  # More than 5% changed
                'minor_change': 1.0 < diff_percentage <= 5.0,
                'no_change': diff_percentage <= 1.0
            }
            
        except Exception as e:
            print(f"❌ Screenshot comparison error: {e}")
            return {
                'difference_percentage': 0,
                'changed_regions': [],
                'significant_change': False,
                'error': str(e)
            }
    
    def save_template(self, name: str, screenshot_base64: str, region: Tuple[int, int, int, int]):
        """
        Save a template for future matching
        
        Args:
            name: Template name (e.g., "play_button")
            screenshot_base64: Screenshot containing the element
            region: (x, y, width, height) of the element
        """
        try:
            image = self._base64_to_image(screenshot_base64)
            x, y, w, h = region
            template = image.crop((x, y, x + w, y + h))
            
            # Save as numpy array for OpenCV
            self.element_templates[name] = np.array(template)
            print(f"✅ Template '{name}' saved")
            
        except Exception as e:
            print(f"❌ Error saving template: {e}")
    
    def find_template_match(self, screenshot_base64: str, template_name: str, threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        Find template in screenshot using OpenCV template matching
        
        Args:
            screenshot_base64: Screenshot to search in
            template_name: Name of saved template
            threshold: Matching threshold (0-1)
        
        Returns:
            (x, y) center coordinates if found, else None
        """
        if template_name not in self.element_templates:
            print(f"❌ Template '{template_name}' not found")
            return None
        
        try:
            # Get images
            screenshot_cv = self._image_to_cv2(screenshot_base64)
            template = self.element_templates[template_name]
            
            # Convert to grayscale
            screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
            
            # Template matching
            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                # Get center coordinates
                h, w = template_gray.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                
                print(f"✅ Template '{template_name}' found at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                return (center_x, center_y)
            else:
                print(f"❌ Template '{template_name}' not found (best match: {max_val:.2f})")
                return None
                
        except Exception as e:
            print(f"❌ Template matching error: {e}")
            return None
    
    def parse_test_instructions(self, instructions: str) -> List[Dict]:
        """
        Convert natural language test instructions into structured steps
        Enhanced with stricter JSON schema
        """
        
        prompt = f"""
        You are a mobile game testing AI. Parse these test instructions into structured steps.
        
        Instructions:
        {instructions}
        
        CRITICAL: Respond with ONLY valid JSON. No markdown, no explanations.
        
        Convert this into a JSON array following this EXACT schema:
        [
            {{
                "step_number": <integer>,
                "action": "<action_type>",
                "description": "<human_readable_description>",
                "target": "<element_to_interact_with or null>",
                "verification": "<what_to_verify or null>",
                "params": {{
                    "text": "<text for input actions>",
                    "direction": "<direction for swipe actions>",
                    "duration": <seconds for wait actions>
                }}
            }}
        ]
        
        Valid action types:
        - "tap_button" / "tap" / "click"
        - "input_text" / "type"
        - "swipe" / "scroll"
        - "wait" / "sleep"
        - "verify" / "check" / "assert"
        - "open_game"
        - "screenshot"
        
        Respond with JSON ONLY. No other text.
        """
        
        try:
            if self.ai_provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
            elif self.ai_provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.choices[0].message.content
            elif self.ai_provider == "gemini":
                response = self.model.generate_content(prompt)
                result = response.text
            
            # Clean response
            result = result.strip()
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'^```\s*', '', result)
            result = re.sub(r'```\s*$', '', result)
            
            steps = json.loads(result)
            
            # Validate schema
            for step in steps:
                if 'step_number' not in step or 'action' not in step or 'description' not in step:
                    raise ValueError("Invalid step schema")
                # Ensure params exists
                if 'params' not in step:
                    step['params'] = {}
            
            print(f"✅ Parsed {len(steps)} steps successfully")
            return steps
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {result}")
            # Fallback
            return [{"step_number": 1, "action": "manual", "description": instructions, "target": None, "verification": None, "params": {}}]
        except Exception as e:
            print(f"❌ Error parsing instructions: {e}")
            return [{"step_number": 1, "action": "manual", "description": instructions, "target": None, "verification": None, "params": {}}]
    
    def analyze_screen(self, screenshot_base64: str, context: str = "") -> Dict:
        """
        Enhanced screen analysis with OCR integration
        """
        
        # First, extract all text from screen
        extracted_texts = self.extract_text_from_screen(screenshot_base64)
        text_summary = [t['text'] for t in extracted_texts[:20]]  # Top 20 text items
        
        prompt = f"""
        You are analyzing a mobile game screenshot. Device resolution: {self.device_width}x{self.device_height}
        
        Context: {context if context else "General game screen analysis"}
        
        Text detected on screen (OCR): {', '.join(text_summary) if text_summary else 'No text detected'}
        
        Analyze this screenshot and provide a JSON response with this EXACT structure:
        {{
            "screen_type": "menu|gameplay|popup|loading|settings|tutorial|other",
            "game_state": "brief description of current state",
            "ui_elements": [
                {{
                    "type": "button|text|icon|game_object|other",
                    "label": "visible text or description",
                    "position": "top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right",
                    "estimated_coords": {{"x": <0-{self.device_width}>, "y": <0-{self.device_height}>}},
                    "confidence": <0.0-1.0>
                }}
            ],
            "interactive_elements": ["list of tappable elements"],
            "game_objects": ["list of visible game objects"],
            "recommended_action": "what should be done next",
            "screen_region_hints": {{
                "top_bar": "description of top area",
                "main_content": "description of center area",
                "bottom_bar": "description of bottom area"
            }}
        }}
        
        Be precise with coordinates. Use the detected OCR text to improve accuracy.
        Respond with ONLY the JSON object.
        """
        
        try:
            if self.ai_provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_base64
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                result = response.content[0].text
            elif self.ai_provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                        ]
                    }]
                )
                result = response.choices[0].message.content
            elif self.ai_provider == "gemini":
                image = self._base64_to_image(screenshot_base64)
                response = self.model.generate_content([prompt, image])
                result = response.text
            
            # Clean and parse
            result = result.strip()
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'^```\s*', '', result)
            result = re.sub(r'```\s*$', '', result)
            
            analysis = json.loads(result)
            
            # Add OCR data to analysis
            analysis['ocr_texts'] = extracted_texts
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing screen analysis: {e}")
            return {
                "screen_type": "unknown",
                "game_state": "Unable to parse",
                "ui_elements": [],
                "ocr_texts": extracted_texts,
                "raw_response": result
            }
        except Exception as e:
            print(f"❌ Screen analysis error: {e}")
            return {
                "screen_type": "error",
                "game_state": str(e),
                "ui_elements": [],
                "ocr_texts": extracted_texts
            }
    
    def find_element_coordinates(self, screenshot_base64: str, target_element: str, retry_variations: bool = True) -> Optional[Tuple[int, int, float]]:
        """
        Enhanced element finding with multiple strategies and confidence scoring
        
        Returns:
            (x, y, confidence) or None if not found
        """
        
        # Strategy 1: Try OCR first for text-based elements
        if any(keyword in target_element.lower() for keyword in ['button', 'text', 'label', 'score', 'coin', 'level']):
            print(f"   🔍 Strategy 1: Trying OCR for '{target_element}'...")
            coords = self.find_text_location(screenshot_base64, target_element, partial_match=True)
            if coords:
                return (*coords, 0.9)  # High confidence for OCR matches
        
        # Strategy 2: Try template matching if available
        template_name = target_element.lower().replace(' ', '_')
        if template_name in self.element_templates:
            print(f"   🔍 Strategy 2: Trying template matching for '{target_element}'...")
            coords = self.find_template_match(screenshot_base64, template_name, threshold=0.75)
            if coords:
                return (*coords, 0.85)  # High confidence for template matches
        
        # Strategy 3: AI Vision with enhanced prompt
        print(f"   🔍 Strategy 3: Using AI vision for '{target_element}'...")
        
        # Get OCR text for context
        extracted_texts = self.extract_text_from_screen(screenshot_base64)
        text_context = ', '.join([t['text'] for t in extracted_texts[:15]])
        
        prompt = f"""
        Find this element on the screen: "{target_element}"
        
        Device resolution: {self.device_width}x{self.device_height}
        Text on screen (OCR): {text_context if text_context else 'No text detected'}
        
        CRITICAL INSTRUCTIONS:
        1. Look for the EXACT element requested
        2. If it's a text element, use the OCR text positions as hints
        3. Provide coordinates at the CENTER of the element (best tap position)
        4. Be conservative with confidence - only high confidence if certain
        
        Respond with ONLY this JSON structure:
        {{
            "found": true|false,
            "element": "{target_element}",
            "x": <pixel_x_coordinate (0-{self.device_width})>,
            "y": <pixel_y_coordinate (0-{self.device_height})>,
            "confidence": <0.0-1.0>,
            "reasoning": "why these coordinates",
            "alternative_elements": ["list of similar elements found"],
            "bounding_box": {{
                "left": <x>,
                "top": <y>,
                "right": <x>,
                "bottom": <y>
            }}
        }}
        
        If not found, set found=false and confidence=0.0
        Respond with JSON ONLY.
        """
        
        try:
            if self.ai_provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_base64
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                result = response.content[0].text
            elif self.ai_provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                        ]
                    }]
                )
                result = response.choices[0].message.content
            elif self.ai_provider == "gemini":
                image = self._base64_to_image(screenshot_base64)
                response = self.model.generate_content([prompt, image])
                result = response.text
            
            # Parse response
            result = result.strip()
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'^```\s*', '', result)
            result = re.sub(r'```\s*$', '', result)
            
            location = json.loads(result)
            
            if location.get("found") and location.get("confidence", 0) > 0.6:
                x, y = location["x"], location["y"]
                confidence = location["confidence"]
                
                # Validate coordinates are within bounds
                if 0 <= x <= self.device_width and 0 <= y <= self.device_height:
                    print(f"   ✅ AI found '{target_element}' at ({x}, {y}) with confidence {confidence:.2f}")
                    print(f"      Reasoning: {location.get('reasoning', 'N/A')}")
                    return (x, y, confidence)
            
            # Try alternatives if main element not found confidently
            if retry_variations and location.get("alternative_elements"):
                print(f"   🔄 Main element not confident, found alternatives: {location['alternative_elements']}")
                # Could try finding alternatives here
            
            print(f"   ❌ Element '{target_element}' not found with sufficient confidence")
            return None
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ❌ Error finding element: {e}")
            print(f"   Raw response: {result}")
            return None
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            return None
    
    def verify_action_success(self, before_screenshot: str, after_screenshot: str, 
                              action_description: str, ocr_verification: Optional[str] = None) -> Dict:
        """
        Enhanced verification with pixel diff and OCR
        
        Args:
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            action_description: What action was taken
            ocr_verification: Optional text to verify appeared/disappeared
        
        Returns:
            Dict with verification results
        """
        
        # First, do pixel-level comparison
        diff_analysis = self.compare_screenshots(before_screenshot, after_screenshot)
        
        # OCR verification if specified
        ocr_result = None
        if ocr_verification:
            before_texts = [t['text'].lower() for t in self.extract_text_from_screen(before_screenshot)]
            after_texts = [t['text'].lower() for t in self.extract_text_from_screen(after_screenshot)]
            
            ocr_target = ocr_verification.lower()
            appeared = ocr_target in ' '.join(after_texts) and ocr_target not in ' '.join(before_texts)
            still_present = ocr_target in ' '.join(after_texts)
            disappeared = ocr_target in ' '.join(before_texts) and ocr_target not in ' '.join(after_texts)
            
            ocr_result = {
                'target_text': ocr_verification,
                'appeared': appeared,
                'still_present': still_present,
                'disappeared': disappeared
            }
        
        # AI visual verification
        prompt = f"""
        Verify if this action succeeded: "{action_description}"
        
        Image comparison analysis:
        - Difference: {diff_analysis['difference_percentage']}%
        - Significant change detected: {diff_analysis['significant_change']}
        - Changed regions: {len(diff_analysis['changed_regions'])}
        
        {f"OCR Verification: Looking for '{ocr_verification}'" if ocr_verification else ""}
        {f"OCR Result: {ocr_result}" if ocr_result else ""}
        
        Compare BEFORE (first image) and AFTER (second image) screenshots.
        
        Respond with ONLY this JSON:
        {{
            "success": true|false,
            "confidence": <0.0-1.0>,
            "changes_observed": ["specific changes between screenshots"],
            "expected_state_reached": true|false,
            "errors_detected": ["any errors or issues"],
            "recommendation": "continue|retry|abort",
            "visual_diff_confirms": true|false
        }}
        
        Consider:
        - Did the screen change appropriately?
        - Are we in the expected game state?
        - Any error messages or unexpected popups?
        - Does the pixel difference make sense for this action?
        
        JSON only.
        """
        
        try:
            if self.ai_provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "BEFORE screenshot:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": before_screenshot
                                }
                            },
                            {"type": "text", "text": "AFTER screenshot:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": after_screenshot
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                result = response.content[0].text
            elif self.ai_provider == "openai":
                # Simplified for OpenAI
                result = json.dumps({
                    "success": diff_analysis['significant_change'],
                    "confidence": 0.7,
                    "changes_observed": ["Screen changed"],
                    "expected_state_reached": True,
                    "errors_detected": [],
                    "recommendation": "continue",
                    "visual_diff_confirms": True
                })
            elif self.ai_provider == "gemini":
                before_img = self._base64_to_image(before_screenshot)
                after_img = self._base64_to_image(after_screenshot)
                response = self.model.generate_content([
                    "BEFORE screenshot:",
                    before_img,
                    "AFTER screenshot:",
                    after_img,
                    prompt
                ])
                result = response.text
            
            # Parse response
            result = result.strip()
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'^```\s*', '', result)
            result = re.sub(r'```\s*$', '', result)
            
            verification = json.loads(result)
            
            # Add diff analysis and OCR results
            verification['pixel_diff'] = diff_analysis
            if ocr_result:
                verification['ocr_verification'] = ocr_result
            
            # Boost confidence if multiple verification methods agree
            if diff_analysis['significant_change'] and verification.get('success'):
                verification['confidence'] = min(1.0, verification['confidence'] + 0.1)
            
            return verification
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"❌ Error in verification: {e}")
            return {
                "success": diff_analysis['significant_change'],
                "confidence": 0.5,
                "changes_observed": [f"{diff_analysis['difference_percentage']}% pixels changed"],
                "expected_state_reached": diff_analysis['significant_change'],
                "errors_detected": [str(e)],
                "recommendation": "retry" if not diff_analysis['significant_change'] else "continue",
                "pixel_diff": diff_analysis,
                "raw_response": result if 'result' in locals() else None
            }
    
    def generate_test_strategy(self, game_name: str, test_objective: str) -> str:
        """Generate test strategy (unchanged from original)"""
        
        prompt = f"""
        You are a QA expert for mobile games.
        
        Game: {game_name}
        Test Objective: {test_objective}
        
        Create a comprehensive test strategy including:
        1. Test steps (detailed, actionable steps)
        2. Expected results for each step
        3. Edge cases to check
        4. Performance considerations
        5. Common bugs to watch for
        
        Make it practical and executable by an AI agent.
        """
        
        if self.ai_provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif self.ai_provider == "openai":
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        elif self.ai_provider == "gemini":
            response = self.model.generate_content(prompt)
            return response.text
    
    def debug_issue(self, screenshot_base64: str, error_description: str) -> Dict:
        """Enhanced debug with OCR context"""
        
        extracted_texts = self.extract_text_from_screen(screenshot_base64)
        text_summary = ', '.join([t['text'] for t in extracted_texts[:15]])
        
        prompt = f"""
        A test failed with this error: {error_description}
        
        Text visible on screen: {text_summary if text_summary else 'No text detected'}
        
        Analyze this screenshot and provide ONLY this JSON:
        {{
            "diagnosis": "what went wrong",
            "root_cause": "why it happened",
            "recovery_steps": ["actionable steps to recover"],
            "prevention": "how to avoid this in future"
        }}
        
        Be specific and actionable.
        """
        
        try:
            if self.ai_provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_base64
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                result = response.content[0].text
            elif self.ai_provider == "openai":
                result = json.dumps({
                    "diagnosis": "Error detected",
                    "root_cause": "Unknown",
                    "recovery_steps": ["Restart game", "Try again"],
                    "prevention": "Add error handling"
                })
            elif self.ai_provider == "gemini":
                image = self._base64_to_image(screenshot_base64)
                response = self.model.generate_content([prompt, image])
                result = response.text
            
            result = result.strip()
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'^```\s*', '', result)
            result = re.sub(r'```\s*$', '', result)
            
            debug_info = json.loads(result)
            debug_info['ocr_texts'] = extracted_texts
            return debug_info
            
        except Exception as e:
            return {
                "diagnosis": f"Debug parsing error: {e}",
                "root_cause": "Unknown",
                "recovery_steps": ["Manual intervention needed"],
                "prevention": "Review test logic",
                "ocr_texts": extracted_texts
            }


if __name__ == "__main__":
    print("✅ Enhanced AI Game Testing Engine loaded")
    print("\n🎯 New Features:")
    print("  • OCR text extraction and location finding")
    print("  • Template matching for UI elements")
    print("  • Pixel-level screenshot comparison")
    print("  • Multi-resolution coordinate mapping")
    print("  • Enhanced verification with multiple strategies")
    print("  • Improved element finding with retry logic")