# 🤖 AI Game Testing Platform

**Test your mobile games using natural language and artificial intelligence!**

The AI sees your game screen, understands what to do, and executes tests automatically - just like a human tester, but powered by AI.

---

## 🎯 What This Does

```
YOU WRITE (in plain English):
┌────────────────────────────────────────┐
│ Open Bus Frenzy                        │
│ Tap Play button                        │
│ Complete Level 1                       │
│ Pick up all passengers                 │
│ Verify coins increased                 │
└────────────────────────────────────────┘
                 ↓
        🤖 AI DOES EVERYTHING
                 ↓
┌────────────────────────────────────────┐
│ ✅ Opened game                         │
│ ✅ Found and tapped Play button        │
│ ✅ Completed Level 1                   │
│ ✅ Collected 5/5 passengers            │
│ ✅ Coins increased from 0 to 100       │
│                                        │
│ 📊 Test Report Generated               │
└────────────────────────────────────────┘
```

---

## ✨ Features

### 🎮 **Test Any Game**
- Casual games (match-3, puzzle, arcade)
- Action games (runners, shooters)
- Strategy games
- Card games
- ANY Android game!

### ✍️ **Natural Language**
- Write tests in plain English
- No coding required
- Just describe what to test

### 🤖 **AI-Powered**
- AI sees the screen like a human
- Finds buttons/elements visually
- Adapts to UI changes
- Handles randomness
- Self-healing tests

### 📊 **Detailed Reports**
- Step-by-step results
- Screenshots for each step
- AI observations
- Error diagnosis
- HTML reports

---

## 🚀 Quick Start

### Prerequisites

1. **Android device** connected via USB
2. **ADB** installed and working
3. **Python 3.8+** installed
4. **AI API key** (Anthropic Claude OR OpenAI GPT-4)

### Installation

```bash
# 1. Clone/download this folder
cd ai_game_tester

# 2. Install dependencies
pip install -r requirements.txt

# 3. Enable USB debugging on your Android device
# Settings → About Phone → Tap "Build Number" 7 times
# Settings → Developer Options → Enable "USB Debugging"

# 4. Connect device and verify ADB works
adb devices
# Should show your device

# 5. Run the dashboard
streamlit run dashboard.py
```

### Get API Key

**Option A: Claude (Recommended)**
1. Go to https://console.anthropic.com/
2. Create account
3. Go to API Keys
4. Create new key
5. Copy it

**Option B: OpenAI GPT-4**
1. Go to https://platform.openai.com/
2. Create account
3. Go to API Keys
4. Create new key
5. Copy it

---

## 📖 How to Use

### Step 1: Connect Device
1. Open the web dashboard (automatically opens after `streamlit run`)
2. Click "Scan for Devices" in sidebar
3. Select your device
4. Click "Connect to Device"
5. ✅ Device connected!

### Step 2: Configure AI
1. In sidebar, select AI provider (Claude or GPT-4)
2. Paste your API key
3. ✅ AI configured!

### Step 3: Write Test
1. Enter your game's package name
   - Example: `com.example.busfrenzy`
   - Find it: Settings → Apps → Your Game → Advanced
2. Write test instructions in natural language
3. Click "Run AI Test"
4. Watch AI execute your test! 🤖

### Step 4: View Results
- See step-by-step execution
- View screenshots
- Check AI observations
- Download HTML report

---

## 💡 Example Test Instructions

### Example 1: Basic Gameplay
```
Open Bus Frenzy game
Skip tutorial if present
Tap "Play" button
Start Level 1
Pick up all passengers
Drive to destination
Complete level
Verify earned 100 coins
```

### Example 2: Login Flow
```
Open the game
Find login button
Tap login
Enter username: test@email.com
Enter password: Test123
Tap submit
Verify logged in successfully
```

### Example 3: Multi-Level Test
```
Open game
Play levels 1 through 5
For each level:
  - Complete all objectives
  - Collect all items
  - Verify completion
Check final score
```

### Example 4: IAP Test
```
Open game
Navigate to shop
Find 100 coins package
Tap to view details
Verify price displayed
Cancel purchase
Go back to menu
```

---

## 🎮 Supported Game Types

### ✅ Works Great With:
- **Casual Games**: Match-3, puzzle, arcade
- **Action Games**: Endless runners, platformers
- **Card Games**: Poker, solitaire, collectible card games
- **Idle Games**: Clicker games, idle simulators
- **Strategy Games**: Tower defense, turn-based strategy

### ⚠️ Limitations:
- Very fast-paced games (requires quick reactions)
- Games requiring precise timing
- Games with complex 3D navigation
- Games requiring multi-touch gestures

---

## 📁 Project Structure

```
ai_game_tester/
├── dashboard.py              # Main web interface
├── ai_engine.py             # AI brain (Claude/GPT-4)
├── device_controller.py      # Device control (ADB)
├── game_test_executor.py     # Test orchestration
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 🔧 Advanced Features

### Custom Templates
Create reusable test templates for common scenarios:
```python
# In dashboard, go to "Use Template" tab
# Select pre-made templates or create your own
```

### Test Multiple Devices
```python
# Connect multiple devices
# Run tests in parallel
# Compare results across devices
```

### CI/CD Integration
```python
# Run tests from command line
python run_test.py --game com.example.game --test test.txt
```

---

## 💰 Cost Estimation

### AI API Costs (per test):

| Provider | Model | Cost per Test | Notes |
|----------|-------|---------------|-------|
| **Anthropic** | Claude 3.5 Sonnet | $0.01-0.05 | Recommended |
| **OpenAI** | GPT-4 Vision | $0.02-0.08 | Alternative |

**Example:**
- 100 tests/month = $1-5
- 1000 tests/month = $10-50

**Free tier available for both providers!**

---

## 🐛 Troubleshooting

### Device Not Found
```bash
# Check ADB connection
adb devices

# Restart ADB if needed
adb kill-server
adb start-server

# Check USB debugging is enabled
# Enable "USB Debugging" in Developer Options
```

### AI Not Working
- ✅ Check API key is correct
- ✅ Check you have credits/quota
- ✅ Try different AI provider
- ✅ Check internet connection

### Test Failing
- 📸 Check screenshots to see what AI sees
- 🤖 AI shows what it's trying to do
- 🔄 AI will retry failed steps automatically
- 📊 Check detailed error in results

### Game Not Starting
```bash
# Check package name is correct
adb shell pm list packages | grep busfrenzy

# Check game is installed
adb shell pm list packages | grep your.game

# Install APK via dashboard if needed
```

---

## 📚 How It Works

### Architecture
```
User (Browser)
    ↓
Streamlit Dashboard (UI)
    ↓
Game Test Executor (Orchestrator)
    ↓
┌──────────────────┬─────────────────┐
│                  │                 │
AI Engine         Device Controller
(Claude/GPT-4)    (ADB Commands)
│                  │
└──────────────────┴─────────────────┘
           ↓
    Android Device
```

### AI Test Flow
1. **Parse**: AI converts natural language to test steps
2. **Analyze**: AI looks at game screenshot
3. **Find**: AI locates buttons/elements visually
4. **Execute**: Device controller taps/swipes/types
5. **Verify**: AI confirms action succeeded
6. **Retry**: If failed, AI tries different approach
7. **Report**: Generate detailed test report

---

## 🎓 Best Practices

### Writing Good Test Instructions

✅ **DO:**
- Be specific: "Tap the red Play button"
- Add verification: "Verify level completed"
- Include expected results: "Should earn 100 coins"
- Break into small steps

❌ **DON'T:**
- Be vague: "Do something with the button"
- Skip verification steps
- Make steps too complex
- Assume AI knows game mechanics

### Example: Good vs Bad

**❌ Bad:**
```
Test the game
```

**✅ Good:**
```
Open Bus Frenzy
Tap Play button
Complete Level 1 by:
  - Picking up 5 passengers
  - Driving to destination
Verify earned 100 coins
Check Level 2 is unlocked
```

---

## 🔐 Security & Privacy

- ✅ All testing happens locally
- ✅ Only screenshots sent to AI (for analysis)
- ✅ No game code or sensitive data sent to AI
- ✅ API keys stored in memory only
- ✅ Screenshots saved locally only

---

## 📊 Metrics & Analytics

The platform tracks:
- Test success rate
- Step-by-step timings
- AI confidence scores
- Screenshot evidence
- Error patterns
- Device performance

---

## 🛠️ Development

### Running Tests
```bash
# Test AI engine
python ai_engine.py

# Test device controller
python device_controller.py

# Test full executor
python game_test_executor.py
```

### Adding Features
- Modify `ai_engine.py` for AI behavior
- Modify `device_controller.py` for device actions
- Modify `dashboard.py` for UI changes

---

## 🤝 Support

### Common Issues
1. **Device not connecting** → Check USB debugging
2. **AI errors** → Check API key and credits
3. **Test timing out** → Increase wait times
4. **Elements not found** → Improve descriptions

### Get Help
- Check troubleshooting section above
- Review example test instructions
- Check AI observations in results
- Try different phrasing for steps

---

## 🎯 Roadmap

### Coming Soon:
- [ ] Multi-device testing
- [ ] Test scheduling
- [ ] Cloud hosting option
- [ ] Pre-recorded test library
- [ ] Performance benchmarking
- [ ] Crash detection
- [ ] Video recording
- [ ] CI/CD integrations

---

## 📝 License

This project is for educational and testing purposes.

---

## 🎉 Success Stories

**"Tested my match-3 game with AI in 5 minutes. Found bugs I missed in manual testing!"** - Indie Developer

**"No more writing Appium scripts. Just describe the test in English!"** - QA Engineer

**"AI adapted when we changed our UI. Tests still work!"** - Game Studio

---

## 🚀 Get Started Now!

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run dashboard.py

# Test!
1. Connect device
2. Enter API key
3. Write test in plain English
4. Watch AI do the work!
```

---

**Happy Testing! 🎮🤖**

Questions? Issues? Just write your problem in natural language and the AI will help debug it! 😉