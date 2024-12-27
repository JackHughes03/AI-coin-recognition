<h1>How to run</h1>
<ul>
  <li>Install imports. Use conda as it seems to work better with tensorflow.</li>
  <li>Download dataset as a zip from <a href="https://www.kaggle.com/datasets/wanderdust/coin-images" target="_blank">here</a></li>
  <li>Unzip and go into coins/data/ and create a new folder called 'dataset'. Put the test and train folders inside of that.</li>
  <li>Copy and paste the dataset folder and cat_to_name.json into the root directory of this project.</li>
  <li>Now run main.py, it will ask to train, test, or validation. Choose train and confirm.<li>
  <li>Let it do it's thing.</li>
  <li>Now to test, go to line 61 and change the path of the test image to whatever you want. Then run main.py and choose test.<li>
</ul>