show_name=/home/jovyan/work/video-subtitle-extractor/data/raw_data
for episode in $show_name/*
do
	echo $episode
        rm -r mylist.txt
	for file in $episode/*.mp4 
	do
		echo $file
		echo "file $file" >> mylist.txt
		#cat tmp/mylist.txt
	done
	base_episode=`basename $episode`
    echo $base_episode
	echo $show_name/$base_episode.mp4
	ffmpeg -f concat -safe 0 -i mylist.txt -c copy $show_name/$base_episode.mp4 -y 1>/dev/null 2>&1
	#ffmpeg -i $show_name/$base_episode.mp4 -ar 44100 -b:a 16k -ac 1 $show_name/$base_episode.wav
	rm -rf $show_name/$base_episode/
	#demucs --two-stems vocals $show_name/$base_episode.mp4 -o ./separated/$show_name	
	#echo $base_episode
	#ffmpeg -i ./separated/$show_name/htdemucs/$base_episode/vocals.wav -ar 16000 -b:a 16k -ac 1 $show_name/$base_episode.wav
done

#ffmpeg -f concat -safe 0 -i mylist.txt -c copy $episode.mp4 -y 1>/dev/null 2>&1
#ffmpeg -i $episoe.mp4 -ar 44100 -b:a 16k -ac 2 $episode.wav
#demucs $episode.wav

