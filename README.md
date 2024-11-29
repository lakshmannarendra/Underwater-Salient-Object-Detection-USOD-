

### **STEPS TO RUN**

Step 1: First, go to the preprocessing folder located in the codes folder. Navigate to the `main.m` file.

Step 2: Modify the directories for your input RGB images that require preprocessing, such as color fusion and enhancement.

Step 3: Save the enhanced images in the output directory specified in `main.m`.

Step 4: Store the outputs as required for the USOD code.

Step 5: Specify the RGB directories for test and train datasets, respectively.

Step 6: Navigate to the directory containing `train_test_eval.py`. Open it and update the paths to your stored pretrained model, trainset, testset, and other required files.

Step 7: Once all directories and requirements for training, testing, and evaluation are set up, run the following command:
```bash
python3 train_test_eval.py --Training 1 --Testing 1 --Evaluation 1 > log/loss.log &
```

Step 8: Monitor the epochs as they start running.

Step 9: To check the last 300 lines of the log, use the following command:
```bash
tail -f -n 300 log/loss.log
```

Step 10: After 200 epochs, the testing procedure begins.

Step 11: At the 60,000th iteration, a checkpoint will be saved as `UVST.pth`.

Step 12: Evaluation metrics will be available in `result.txt`.

Step 13: In the zip file `30_dl_code`, I have included preprocessing and `USOD_10k` code. Preprocessing comprises color enhancement Matlab code, while `USOD_10k` contains datasets, pretrained models, and Python files for evaluation, testing, and training.

## Contributors

Thanks to these wonderful people for their contributions:


- [lakshmannarendra](https://github.com/lakshmannarendra)
- [apraneeth20](https://github.com/apraneeth20)
  

---

