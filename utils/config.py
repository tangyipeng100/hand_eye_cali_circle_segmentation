#hand_eye_infer.py parameters
dataset_pre = './data/Standard_sphere_seg_dataset_v1/'   # Standard_sphere_seg_dataset_v1 path, which has been defaultly put in ./data
model_path = './checkpoints/PointNet_64.pkl'
vis_file_name = 'label_out_1/60_noise.csv'   #  select a sample and show its inference result, for opt.mode == 'infer_results'
test_file_name = 'test_set/test_file.txt'  # test set sample file dictionaries
points_n = 1280
output_base_dir = './output_file'
infer_batch_size = 36

#ransac_circle_fitting.py parameters
output_save_dir = './log'
ref_ball_r = 19.05   # calibration standard sphere radius
fitting_circle_name = 'average_center.csv'
