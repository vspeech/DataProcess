for i in `cat ./list_ids`
do
	echo $i
	showname=$i
	rm -rf data/htdemucs
	rm -rf data/raw_data/*
	ossutil cp oss://aigcdevbj/USERS/xn/dataset/vc_train/$showname/ data/raw_data/ --recursive	
	sh data/concat.sh
	#python paddle_ocr_demucs_ffmpeg_multi-fast.py --recipe RECIPE
	#python merge_timeline_all.py
	#python spk_cluster.py $showname

	#ossutil cp data/output_data/$showname oss://aigcdevbj/USERS/xn/dataset/vc_train/audio/$showname/
done
#showname=$1
#rm -rf data/htdemucs
#rm -rf data/raw_data/*
#ossutil cp oss://aigcdevbj/USERS/dexiao/test_upload/$showname/ data/raw_data/ --recursive
#sh data/concat.sh
#python paddle_ocr_demucs_ffmpeg_multi-fast.py
#python merge_timeline_all.py
#python spk_cluster.py $showname 
