# Mobile Video Compatibility Fix

This document explains the mobile compatibility issues with yt-dlp downloads and the solutions implemented.

## Problem

Videos downloaded by yt-dlp were playing correctly on desktop but failing to play on mobile devices. This is typically caused by:

1. **Incompatible video codecs**: VP9, AV1, or high H.264 profiles not supported on older mobile devices
2. **Incompatible audio codecs**: Opus, Vorbis, or other codecs not natively supported on mobile
3. **Wrong container format**: WebM containers not universally supported on mobile
4. **Missing faststart flag**: Videos not optimized for progressive download/streaming
5. **High H.264 profiles**: High/High10 profiles requiring more processing power than available on mobile

## Solution

### 1. Updated yt-dlp Configuration

The server has been updated with three mobile-optimized download configurations:

#### Mobile Optimized (Primary)
```javascript
{
  name: 'mobile_optimized',
  args: [
    '--format', 'best[height<=720][vcodec*=avc1][acodec*=mp4a]/best[height<=1080][vcodec*=avc1][acodec*=mp4a]/best[vcodec*=avc1][acodec*=mp4a]',
    '--merge-output-format', 'mp4',
    '--postprocessor-args', 'ffmpeg:-c:v libx264 -profile:v baseline -level 3.1 -c:a aac -ar 44100 -ac 2 -movflags +faststart',
    '--prefer-ffmpeg',
    '--no-warnings',
    '--ignore-errors',
    '--extract-flat', 'false',
    '--no-playlist',
    '-o', tempVideoPath,
    '--progress',
    '--', youtubeUrl
  ]
}
```

**Features:**
- Prioritizes H.264 (avc1) video codec with AAC (mp4a) audio
- Uses baseline profile for maximum mobile compatibility
- Limits resolution to 720p/1080p for reasonable file sizes
- Adds faststart flag for progressive download

#### Mobile Compatible H.264 (Fallback)
```javascript
{
  name: 'mobile_compatible_h264',
  args: [
    '--format', '18/22/best[ext=mp4][vcodec*=avc1]/bestvideo[ext=mp4][vcodec*=avc1]+bestaudio[ext=m4a]/best',
    '--merge-output-format', 'mp4',
    '--postprocessor-args', 'ffmpeg:-c:v libx264 -profile:v main -level 3.1 -preset medium -crf 23 -c:a aac -b:a 128k -ar 44100 -ac 2 -movflags +faststart',
    '--prefer-ffmpeg',
    '--no-warnings',
    '-o', tempVideoPath,
    '--progress',
    '--', youtubeUrl
  ]
}
```

**Features:**
- Uses specific YouTube format IDs (18=360p, 22=720p) known to be mobile-compatible
- Main profile H.264 for good quality/compatibility balance
- 128k AAC audio at 44.1kHz (standard mobile audio)

#### Legacy Mobile (Final Fallback)
```javascript
{
  name: 'legacy_mobile',
  args: [
    '--format', 'worst[height>=360][acodec!=none]/best[height<=480][acodec!=none]/best[acodec!=none]',
    '--merge-output-format', 'mp4',
    '--postprocessor-args', 'ffmpeg:-c:v libx264 -profile:v baseline -level 3.0 -maxrate 1000k -bufsize 2000k -c:a aac -b:a 96k -ar 44100 -ac 2 -movflags +faststart',
    '--prefer-ffmpeg',
    '--no-warnings',
    '-o', tempVideoPath,
    '--progress',
    '--', youtubeUrl
  ]
}
```

**Features:**
- Conservative settings for older mobile devices
- Baseline profile with level 3.0 (widely supported)
- Lower bitrates (1000k video, 96k audio) for bandwidth-constrained devices
- Minimum 360p resolution with maximum 480p

### 2. Mobile Video Optimizer Utility

A new utility (`src/mobile_video_optimizer.js`) provides:

#### Functions:
- `checkMobileCompatibility(filePath)` - Analyze video compatibility
- `optimizeForMobile(inputPath, outputPath, options)` - Re-encode videos for mobile
- `getOptimalMobileFormat(url)` - Get best mobile format for a YouTube URL
- `analyzeMobileCompatibility(mediaInfo)` - Detailed compatibility analysis

#### Key Features:
- **Codec Detection**: Identifies incompatible video/audio codecs
- **Profile Analysis**: Checks H.264 profile compatibility
- **Container Validation**: Ensures MP4/MOV container usage
- **Bitrate Analysis**: Reviews bitrates for mobile optimization
- **Automatic Re-encoding**: Converts videos to mobile-compatible formats

### 3. Test Script

The `test_mobile_compatibility.js` script provides command-line tools:

```bash
# Check if a video file is mobile-compatible
node test_mobile_compatibility.js check video.mp4

# Download a YouTube video with mobile optimization
node test_mobile_compatibility.js download "https://youtube.com/watch?v=VIDEO_ID"

# Re-encode an existing video for mobile compatibility
node test_mobile_compatibility.js optimize input.mp4 output_mobile.mp4

# Show available formats and mobile recommendation
node test_mobile_compatibility.js formats "https://youtube.com/watch?v=VIDEO_ID"
```

## Mobile Compatibility Requirements

### Video Codec Requirements:
- **Codec**: H.264/AVC (most widely supported)
- **Profile**: Baseline or Main (avoid High/High10 profiles)
- **Level**: 3.0 or 3.1 (hardware decoding support)
- **Container**: MP4 (universal mobile support)

### Audio Codec Requirements:
- **Codec**: AAC (native mobile support)
- **Sample Rate**: 44.1kHz or 48kHz
- **Channels**: Stereo (2 channels)
- **Bitrate**: 96k-128k (good quality/size balance)

### Container Requirements:
- **Format**: MP4 container
- **Faststart**: Enabled for progressive download
- **Moov atom**: At beginning of file for streaming

## Usage

### For New Downloads:
The server automatically uses mobile-optimized configurations when downloading YouTube videos through the `/process/youtube-url` endpoint.

### For Existing Videos:
Use the test script to check and optimize existing videos:

```bash
# Check compatibility
node test_mobile_compatibility.js check /path/to/video.mp4

# Optimize if needed
node test_mobile_compatibility.js optimize /path/to/video.mp4 /path/to/mobile_optimized.mp4
```

### Manual yt-dlp Usage:
For manual downloads, use the mobile-optimized format string:

```bash
yt-dlp -f "best[height<=720][vcodec*=avc1][acodec*=mp4a]/18/22/best" \
  --merge-output-format mp4 \
  --postprocessor-args "ffmpeg:-c:v libx264 -profile:v main -level 3.1 -c:a aac -ar 44100 -ac 2 -movflags +faststart" \
  "https://youtube.com/watch?v=VIDEO_ID"
```

## Testing

### Test on Mobile Devices:
1. Download a video using the new configuration
2. Transfer to mobile device or upload to a web server
3. Test playback on various mobile browsers and native video players
4. Verify smooth playback without buffering issues

### Compatibility Matrix:
| Device Type | H.264 Baseline | H.264 Main | H.264 High | AAC Audio | MP4 Container |
|-------------|----------------|------------|------------|-----------|---------------|
| iOS Safari  | ✅ | ✅ | ✅ | ✅ | ✅ |
| Android Chrome | ✅ | ✅ | ⚠️* | ✅ | ✅ |
| Older Android | ✅ | ⚠️* | ❌ | ✅ | ✅ |

*May require hardware decoding support

## Troubleshooting

### Videos Still Not Playing on Mobile:
1. Check the video with the compatibility checker:
   ```bash
   node test_mobile_compatibility.js check video.mp4
   ```

2. Re-optimize the video:
   ```bash
   node test_mobile_compatibility.js optimize video.mp4 video_mobile.mp4
   ```

3. Use more conservative settings for very old devices:
   ```javascript
   const options = {
     videoProfile: 'baseline',
     videoLevel: '3.0',
     maxWidth: 854,
     maxHeight: 480,
     audioBitrate: '96k'
   };
   ```

### yt-dlp Installation Issues:
Ensure yt-dlp is installed and up to date:
```bash
pip install --upgrade yt-dlp
```

### FFmpeg Issues:
Ensure FFmpeg is installed with H.264 and AAC support:
```bash
ffmpeg -encoders | grep -E "(libx264|aac)"
```

## Performance Considerations

### File Size:
- Mobile-optimized videos are typically 20-40% smaller than high-quality downloads
- Baseline profile adds ~5-10% size overhead vs High profile
- Lower resolution (720p vs 1080p) significantly reduces file size

### Quality:
- Main profile provides good quality/compatibility balance
- Baseline profile maximizes compatibility but slightly reduces quality
- CRF 23 provides good quality for most content types

### Bandwidth:
- Mobile-optimized videos load faster on cellular connections
- Progressive download (faststart) enables immediate playback
- Lower bitrates reduce buffering on slow connections

## Future Improvements

### Potential Enhancements:
1. **Adaptive Bitrate**: Multiple quality options for different connection speeds
2. **Device Detection**: Automatic format selection based on User-Agent
3. **Background Optimization**: Automatic re-encoding of existing library
4. **Quality Metrics**: VMAF/SSIM analysis for quality validation
5. **HDR Support**: Mobile-compatible HDR encoding for supported devices 