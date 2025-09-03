# 🚀 Quick Test Guide - Real-Time Features

## 📋 **Checklist Before Testing**

### ✅ **Dependencies Installed?**
```bash
pip install mediapipe opencv-python numpy
```

### ✅ **Server Running?**
```bash
# From multimodal-assistant directory
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

### ✅ **Camera Available?**
- Webcam connected and working
- Browser permissions for camera access
- Good lighting condition

---

## 🎯 **3-Minute Quick Test**

### **1. Health Check (30 seconds)**
```bash
# Test 1: API Health
curl http://localhost:8000/health
# Expected: {"ok": true, "status": "healthy", "version": "1.0.0"}

# Test 2: Swagger UI
open http://localhost:8000/docs
# Expected: Interactive API documentation
```

### **2. Web Demo (2 minutes)**
```bash
# Open real-time demo
open http://localhost:8000/demo
```

**In Browser:**
1. Click **"Start Camera"** 
2. Allow camera permissions
3. Click **"Start Gesture"** 
4. Show hand gestures:
   - ✊ **Fist** (close all fingers)
   - 👍 **Thumbs Up** (only thumb extended)  
   - ✌️ **Peace** (index + middle fingers)
   - 🖐️ **Open Palm** (all fingers extended)

5. Click **"Start Eye Tracking"**
6. Look at camera and:
   - Look **left, right, center**
   - **Blink** normally
   - Watch live statistics

### **3. Test Results (30 seconds)**
**Expected Outputs:**
- ✅ Real-time gesture detection with confidence scores
- ✅ Live gaze direction tracking (Left/Right/Center)
- ✅ Blink detection with EAR values
- ✅ Statistics: gesture counts, blink count, FPS
- ✅ Processed images with landmarks overlay

---

## 🔧 **WebSocket Testing (Advanced)**

### **Test with JavaScript Console:**
```javascript
// Test gesture recognition WebSocket
const gestureWS = new WebSocket('ws://localhost:8000/v1/realtime/gesture');

gestureWS.onopen = () => console.log('Gesture WS connected');
gestureWS.onmessage = (event) => console.log('Gesture:', JSON.parse(event.data));

// Test eye tracking WebSocket  
const eyeWS = new WebSocket('ws://localhost:8000/v1/realtime/eyetracking');

eyeWS.onopen = () => console.log('Eye WS connected');
eyeWS.onmessage = (event) => console.log('Eye:', JSON.parse(event.data));
```

### **Send Test Image:**
```javascript
// Capture from video element and send
const video = document.querySelector('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
ctx.drawImage(video, 0, 0);

const imageData = canvas.toDataURL('image/jpeg');

gestureWS.send(JSON.stringify({
    type: 'image',
    data: imageData
}));
```

---

## 🐛 **Troubleshooting**

### **❌ Camera Not Working**
```bash
# Check browser permissions
# Chrome: Settings → Privacy → Camera
# Firefox: Preferences → Privacy → Permissions → Camera
```

### **❌ WebSocket Connection Failed**
```bash
# Check server is running
netstat -an | grep 8000

# Check CORS settings (already configured)
# Restart server if needed
```

### **❌ Low Performance/FPS**
```bash
# Reduce video resolution in demo
# Close other applications using camera
# Use Chrome browser (better WebRTC performance)
```

### **❌ Gesture Recognition Inaccurate**
```bash
# Ensure good lighting
# Keep hand 2-3 feet from camera
# Make clear, distinct gestures
# Wait for model to stabilize (2-3 seconds)
```

### **❌ Eye Tracking Not Working**
```bash
# Face camera directly
# Ensure eyes are clearly visible
# Remove glasses if causing issues
# Improve lighting on face
```

---

## 📊 **Performance Benchmarks**

### **Expected Performance:**
- **FPS**: 10-15 FPS (web demo)
- **Latency**: <100ms processing time
- **Accuracy**: 
  - Gesture recognition: >90% with clear gestures
  - Eye gaze: >85% accuracy
  - Blink detection: >95% accuracy

### **System Requirements:**
- **CPU**: Modern processor (Intel i5+ or equivalent)
- **RAM**: 4GB+ available
- **Camera**: 720p+ webcam
- **Browser**: Chrome 80+, Firefox 75+, Safari 13+

---

## 🎮 **Fun Test Scenarios**

### **Gesture Control Demo:**
1. Use gestures to control music:
   - 👍 = Play/Pause
   - ✊ = Stop
   - ✌️ = Next track
   - 🖐️ = Volume up

### **Eye Gaze Navigation:**
1. Look at different UI elements
2. Use blinks as "clicks"
3. Navigate menus with eye movement

### **Attention Monitoring:**
1. Simulate studying/working session
2. Track blink rate and fatigue
3. Monitor gaze patterns

---

## 📈 **Next Steps After Testing**

### **Integration Ideas:**
- Add gesture controls to existing apps
- Build accessibility interfaces
- Create interactive presentations
- Develop gaming controls
- Build attention monitoring systems

### **Extend Features:**
- Add more gesture types
- Implement face recognition
- Add pose estimation
- Build custom training pipelines
- Create mobile app interface

---

## 🎯 **Success Criteria**

**You've successfully tested when:**
- ✅ Camera feed appears in browser
- ✅ Hand gestures are detected with >80% confidence
- ✅ Eye gaze direction is tracked accurately
- ✅ Blinks are detected consistently
- ✅ WebSocket connections work smoothly
- ✅ FPS is stable (>8 FPS)
- ✅ No errors in browser console

**🎉 Congratulations! Your real-time vision system is working!**

---

**Need help?** Check the detailed documentation in `REALTIME_DEMOS.md`

