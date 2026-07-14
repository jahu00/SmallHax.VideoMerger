# VideoMerger
This is an app for syncing audio between 2 versions of the same video.

Audio will be sped up/slowed down and padded/trimmed to match the other video.
It's a 2 pass process, but the app will try to do padding/trimming without reencoding.
You can both keep or change pitch when changing speed.
There is also optional audio normalization.

App is made ine Python and uses TKInter for UI. tkinterdnd2 is optional and olny needed for drag and drop functionality.

App relies on ffmpeg being already present on the system.

App only produces audio files, you have to mux them yourself (for example using MKVToolNix).

## Typical use case
Sync audio in one language from a DVD rip with video from a BD rip to mux later in mp4 container for use with jellyfin.
While jellyfin is very felxible, h264+aac in mp4 format causes the least amount of trouble (either as direct playback or simple remux).
It even works with multiple audio streams (but fails if audio streams are in a separate file).
Padding/trimming is also important, because stream delay from container rarely works with direct playback.

## How to use
- Load both videos
- Find the same frame at the start of the video (preferibly the first frame of a scene) and mark it
- Do the same thing near the end of the video
- Select the right audio track (if video has more than one)
- With 2 markers set on each video, click one of the save buttons

## Known issues
This method assumes the audio is already in sync in the source file.
It will not work if the audio that is to be synced has custom timing (and possibly also delay) set in the source container.
