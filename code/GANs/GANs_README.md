## Steps to Use GANimation for Image Modification

1. **Unzip the Data File**
   - Navigate to the directory `./ganimation_replicate/datasets/celebA`.
   - Unzip the file `aus_openface.pkl.zip`.

2. **Download the Model**
   - Download the model from the provided Google Drive link : (https://drive.google.com/drive/folders/1bQb8Nun2ijjknIdYemaHwa7qJVnjFpQQ?usp=sharing)
   - Add the downloaded model to the `./ganimation_replicate` folder.

3. **Prepare the Images**
   - Place the two images you want to modify in the `./ganimation_replicate/datasets/celebA/imgs` folder.
   - Update the file `./ganimation_replicate/datasets/celebA/test_ids.csv` with the names of these image files.

4. **Run the Codebase**
   - Use the following command to run the model:
     ```bash
     python main.py --mode test --data_root ganimation_replicate/datasets/celebA --ckpt_dir [path_to_model] --load_epoch [epoch_num] --model stargan
     ```
     For model flag, you can also use ganimation, but stargan gives better results

5. **Check the Output**
   - Find the modified images in the `results` folder.

6. **Reference**
   - The implementation is based on the GitHub repository: [donydchen/ganimation_replicate](https://github.com/donydchen/ganimation_replicate/tree/master)
