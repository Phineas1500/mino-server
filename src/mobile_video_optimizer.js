const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Mobile Video Optimizer Utility
 * 
 * This utility helps ensure downloaded videos are compatible with mobile devices
 * by checking codecs and providing re-encoding options when needed.
 */

/**
 * Check if a video file is mobile-compatible
 * @param {string} filePath - Path to the video file
 * @returns {Promise<Object>} Compatibility analysis
 */
async function checkMobileCompatibility(filePath) {
  return new Promise((resolve, reject) => {
    const ffprobe = spawn('ffprobe', [
      '-v', 'quiet',
      '-print_format', 'json',
      '-show_format',
      '-show_streams',
      filePath
    ]);

    let output = '';
    let errorOutput = '';

    ffprobe.stdout.on('data', (data) => {
      output += data.toString();
    });

    ffprobe.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    ffprobe.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`FFprobe failed: ${errorOutput}`));
      }

      try {
        const info = JSON.parse(output);
        const analysis = analyzeMobileCompatibility(info);
        resolve(analysis);
      } catch (error) {
        reject(error);
      }
    });
  });
}

/**
 * Analyze video/audio streams for mobile compatibility
 * @param {Object} mediaInfo - FFprobe output
 * @returns {Object} Detailed compatibility analysis
 */
function analyzeMobileCompatibility(mediaInfo) {
  const videoStream = mediaInfo.streams.find(s => s.codec_type === 'video');
  const audioStream = mediaInfo.streams.find(s => s.codec_type === 'audio');
  
  const analysis = {
    isCompatible: true,
    issues: [],
    recommendations: [],
    details: {
      video: null,
      audio: null,
      container: null
    }
  };

  // Analyze container format
  const containerFormat = mediaInfo.format.format_name.toLowerCase();
  analysis.details.container = {
    format: containerFormat,
    compatible: containerFormat.includes('mp4') || containerFormat.includes('mov')
  };

  if (!analysis.details.container.compatible) {
    analysis.isCompatible = false;
    analysis.issues.push('Container format not mobile-friendly');
    analysis.recommendations.push('Convert to MP4 container');
  }

  // Analyze video stream
  if (videoStream) {
    const videoCodec = videoStream.codec_name.toLowerCase();
    const profile = videoStream.profile ? videoStream.profile.toLowerCase() : '';
    const level = videoStream.level || 0;
    
    analysis.details.video = {
      codec: videoCodec,
      profile: profile,
      level: level,
      width: videoStream.width,
      height: videoStream.height,
      bitRate: parseInt(videoStream.bit_rate) || 0
    };

    // Check video codec compatibility
    if (videoCodec !== 'h264') {
      analysis.isCompatible = false;
      analysis.issues.push(`Video codec '${videoCodec}' may not be supported on all mobile devices`);
      analysis.recommendations.push('Re-encode to H.264');
    }

    // Check H.264 profile compatibility
    if (videoCodec === 'h264' && profile && !['baseline', 'main', 'constrained baseline'].includes(profile)) {
      analysis.isCompatible = false;
      analysis.issues.push(`H.264 profile '${profile}' may not be compatible with older mobile devices`);
      analysis.recommendations.push('Re-encode with baseline or main profile');
    }

    // Check resolution for mobile optimization
    if (videoStream.width > 1920 || videoStream.height > 1080) {
      analysis.recommendations.push('Consider reducing resolution to 1080p or lower for better mobile performance');
    }
  }

  // Analyze audio stream
  if (audioStream) {
    const audioCodec = audioStream.codec_name.toLowerCase();
    const sampleRate = parseInt(audioStream.sample_rate) || 0;
    const channels = audioStream.channels || 0;
    
    analysis.details.audio = {
      codec: audioCodec,
      sampleRate: sampleRate,
      channels: channels,
      bitRate: parseInt(audioStream.bit_rate) || 0
    };

    // Check audio codec compatibility
    if (!['aac', 'mp3'].includes(audioCodec)) {
      analysis.isCompatible = false;
      analysis.issues.push(`Audio codec '${audioCodec}' may not be supported on mobile devices`);
      analysis.recommendations.push('Re-encode audio to AAC');
    }

    // Check sample rate
    if (sampleRate > 48000) {
      analysis.recommendations.push('Consider reducing audio sample rate to 44.1kHz or 48kHz');
    }
  }

  return analysis;
}

/**
 * Re-encode video for mobile compatibility
 * @param {string} inputPath - Input video file path
 * @param {string} outputPath - Output video file path
 * @param {Object} options - Encoding options
 * @returns {Promise<void>}
 */
async function optimizeForMobile(inputPath, outputPath, options = {}) {
  const defaultOptions = {
    videoCodec: 'libx264',
    videoProfile: 'main',
    videoLevel: '3.1',
    videoCRF: 23,
    audioCodec: 'aac',
    audioBitrate: '128k',
    audioSampleRate: 44100,
    audioChannels: 2,
    maxWidth: 1920,
    maxHeight: 1080,
    preset: 'medium'
  };

  const settings = { ...defaultOptions, ...options };

  return new Promise((resolve, reject) => {
    const ffmpegArgs = [
      '-i', inputPath,
      '-c:v', settings.videoCodec,
      '-profile:v', settings.videoProfile,
      '-level:v', settings.videoLevel,
      '-crf', settings.videoCRF.toString(),
      '-preset', settings.preset,
      '-vf', `scale='min(${settings.maxWidth},iw)':'min(${settings.maxHeight},ih)':force_original_aspect_ratio=decrease`,
      '-c:a', settings.audioCodec,
      '-b:a', settings.audioBitrate,
      '-ar', settings.audioSampleRate.toString(),
      '-ac', settings.audioChannels.toString(),
      '-movflags', '+faststart',
      '-y', // Overwrite output file
      outputPath
    ];

    console.log('Starting mobile optimization with command:', 'ffmpeg', ffmpegArgs.join(' '));

    const ffmpeg = spawn('ffmpeg', ffmpegArgs);

    let errorOutput = '';

    ffmpeg.stderr.on('data', (data) => {
      const text = data.toString();
      errorOutput += text;
      
      // Parse progress if needed
      const progressMatch = text.match(/time=(\d{2}:\d{2}:\d{2}\.\d{2})/);
      if (progressMatch) {
        console.log('Encoding progress:', progressMatch[1]);
      }
    });

    ffmpeg.on('close', (code) => {
      if (code === 0) {
        console.log('Mobile optimization completed successfully');
        resolve();
      } else {
        console.error('FFmpeg error output:', errorOutput);
        reject(new Error(`FFmpeg failed with code ${code}: ${errorOutput}`));
      }
    });

    ffmpeg.on('error', (error) => {
      reject(new Error(`FFmpeg spawn error: ${error.message}`));
    });
  });
}

/**
 * Get optimal yt-dlp format string for mobile compatibility
 * @param {string} url - YouTube URL
 * @returns {Promise<string>} Recommended format string
 */
async function getOptimalMobileFormat(url) {
  return new Promise((resolve, reject) => {
    const ytdlp = spawn('yt-dlp', ['-F', '--', url]);
    
    let output = '';
    let errorOutput = '';

    ytdlp.stdout.on('data', (data) => {
      output += data.toString();
    });

    ytdlp.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    ytdlp.on('close', (code) => {
      if (code !== 0) {
        console.warn(`yt-dlp format listing failed: ${errorOutput}`);
        // Return a robust fallback format that should work for most videos
        resolve('best[height<=720][vcodec*=avc1][acodec*=mp4a]/best[height<=720][ext=mp4]/best[ext=mp4]');
        return;
      }

      try {
        // Parse available formats and recommend the best mobile-compatible one
        const mobileFormat = parseMobileCompatibleFormats(output);
        resolve(mobileFormat);
      } catch (error) {
        console.warn(`Error parsing formats: ${error.message}`);
        // Return fallback format
        resolve('best[height<=720][vcodec*=avc1][acodec*=mp4a]/best[height<=720][ext=mp4]/best[ext=mp4]');
      }
    });

    ytdlp.on('error', (error) => {
      console.warn(`yt-dlp spawn error: ${error.message}`);
      // Return fallback format
      resolve('best[height<=720][vcodec*=avc1][acodec*=mp4a]/best[height<=720][ext=mp4]/best[ext=mp4]');
    });
  });
}

/**
 * Parse yt-dlp format list and return mobile-compatible format string
 * @param {string} formatList - yt-dlp -F output
 * @returns {string} Optimal format string for mobile
 */
function parseMobileCompatibleFormats(formatList) {
  const lines = formatList.split('\n');
  
  // Check what types of formats are available
  let hasAvc1Video = false;
  let hasAudioStreams = false;
  let hasMp4Format = false;
  
  for (const line of lines) {
    if (line.includes('avc1') && line.includes('video only')) {
      hasAvc1Video = true;
    }
    if (line.includes('audio only')) {
      hasAudioStreams = true;
    }
    if (line.includes('mp4') && !line.includes('video only') && !line.includes('audio only')) {
      hasMp4Format = true;
    }
  }
  
  console.log(`Format analysis: hasAvc1Video=${hasAvc1Video}, hasAudioStreams=${hasAudioStreams}, hasMp4Format=${hasMp4Format}`);
  
  // Use progressively simpler format strings
  if (hasAvc1Video && hasAudioStreams) {
    // Try H.264 video with separate audio first
    return 'bestvideo[vcodec*=avc1][height<=720]+bestaudio/best[height<=720]/best';
  } else if (hasMp4Format) {
    // Fall back to any mp4 format
    return 'best[ext=mp4][height<=720]/best[height<=720]/best';
  } else {
    // Ultimate fallback - just get the best available
    return 'best[height<=720]/best';
  }
}

module.exports = {
  checkMobileCompatibility,
  optimizeForMobile,
  getOptimalMobileFormat,
  analyzeMobileCompatibility
}; 