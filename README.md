# IRL MOVEMENT REPLICATION
Training a unitree g1-humanoid to replicate a movement from a video.

HUMANOID LINK: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1

VIDEO LINK: https://www.youtube.com/shorts/q3KCMEi6Khc

## PROCESS
1. downloaded frames from the video using _yt-dlp_
2. Used mediapipe to get landmarks which is 33 keypoints(positions of parts in space).
2. Retargetting the mediapipe landmarks to G1-humanoid joints angles 
3. Then using those joint angles we generate expert trajectories which act as the Teacher for the humanoid to learn movement.
5. Experimented with GAIL and Behaviour Cloning Methods for learning movement 
6. BehaviourC did way better than GAIL, again could be due faulty tuning of hyperparameters.
7. It was not the best replication of movement but it did show promise 


## USAGE 
1. Download the video using `yt-dlp -f mp4 <VIDEO LINK>`
2. Keep the video in the Video folder 
3. download the assets the g1.xml from the humanoid link and also assets
4. Then first run get_poses.py to get the landmarks
5. then run target to retarget the poses to humanoid joints 
6. Then train.py to train it or just test it. 




