# Data Mining and Label Generation Pipeline

This function is designed to mine customized data from the PLUSAI dataset and utilize an open-source detection model to generate labels. The process involves three main steps: first, collecting valid clips from the PLUSAI dataset by reading the `clip_info.json` file. To start the collection process, run the following command: 

```bash
python zj_apps/data_miner/mine_non_vehicle_clips.py --config-yaml zj_apps/cfg/non_vehicle_data.yaml
```

Next, prelabeling is performed by detecting customized objects within the clips and generating YOLO format labels. You can start prelabeling by running this command:

```
bash zj_apps/dino/detect_dir.sh --config-yaml zj_apps/cfg/non_vehicle_data.yaml
```

Finally, after prelabeling, a human check is carried out to validate the detection results. The images are visualized, and if the labels are correct, the human annotator will mark them as "pass."

This should be what you're looking for. You can now copy everything in one go!