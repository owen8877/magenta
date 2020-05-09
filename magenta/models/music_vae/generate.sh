python music_vae_generate.py \
--config=hierdec-mel_16bar \
--checkpoint_file=checkpoint/hierdec-mel_16bar.tar \
--mode=interpolate \
--num_outputs=5 \
--input_midi_1=/home/xdroid/Shared/dataset/test-2/export-1.mid \
--input_midi_2=/home/xdroid/Shared/dataset/test-2/export-2.mid \
--output_dir=/tmp/music_vae/generated