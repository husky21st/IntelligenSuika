#!/bin/sh

mkdir movies
 ./yt-dlp --match-title "【スイカゲーム】" -f 136 -o './movies/%(title)s.%(ext)s' $1

# ./yt-dlp --match-title "【スイカゲーム】" -f "[ext=mp4][resolution=1280x720][fps=30]" -o './movies/%(title)s.%(ext)s' $1
