# -Training-Word-Vectors-with-Facebook-s-fastText
To train word vectors using Facebook's fastText, you can use the Python fasttext library. Below is a step-by-step guide to training word vectors using fastText.
Prerequisites:

    Install the necessary libraries:

    pip install fasttext

    Prepare your dataset:
        You need a text file where each line contains a sentence or document. fastText will learn word vectors based on the context in which the words appear. For this example, we will use a simple text file (your_dataset.txt).

    Training the Word Vectors:
        We'll train the word vectors using skip-gram (predicting context from a target word) or CBOW (predicting a target word from context) models. By default, fastText uses the skip-gram model.

Example Python Code to Train Word Vectors:

import fasttext

# Step 1: Prepare a simple text file with sentences for training (your_dataset.txt)
# For example, your dataset can be:
# "This is a sample sentence."
# "Another sentence for training word vectors."
# "fastText is a powerful library for text representation."
# Make sure your dataset file is large enough for meaningful word embeddings.

# Step 2: Train the Word Vectors using fastText
# Use the `fasttext.train_unsupervised` method to train a model

model = fasttext.train_unsupervised('your_dataset.txt', model='skipgram')  # You can change model to 'cbow' for Continuous Bag of Words

# Step 3: Save the trained model
model.save_model("word_vectors.bin")

# Step 4: Get word vectors for a specific word
word_vector = model.get_word_vector("sample")  # Replace with any word in your dataset

print(f"Word vector for 'sample': {word_vector}")

# Step 5: Find similar words to a given word
# You can find the most similar words to any word using the `get_nearest_neighbors` method
similar_words = model.get_nearest_neighbors("sample")  # Replace with any word
print(f"Most similar words to 'sample': {similar_words}")

# Step 6: Using the model for text classification (optional)
# You can also fine-tune fastText for supervised tasks like text classification.
# Example: model = fasttext.train_supervised('training_data.txt')

Key Points:

    fasttext.train_unsupervised: This is used to train the word vectors. It has several options like model='skipgram' (default) and model='cbow'. You can experiment with both to see which gives better results for your data.
    model.get_word_vector(word): This method returns the word vector for a given word.
    model.get_nearest_neighbors(word): This method returns the nearest neighbors (most similar words) to a given word based on cosine similarity in the vector space.

Example Output:

If your dataset contains the word "sample", the output might look something like this:

Word vector for 'sample': [ 0.12345, -0.09876, ...]  # Vector representation of the word

Most similar words to 'sample': [(0.85, 'example'), (0.78, 'demo'), (0.75, 'sampled'), ...]  # Similar words with cosine similarity

Options and Parameters for train_unsupervised:

    model: This parameter can be either 'skipgram' or 'cbow' (Continuous Bag of Words). Skipgram is usually better for small datasets or rare words, while CBOW is more efficient with large datasets.
    lr: The learning rate for the model. Default is 0.05.
    dim: Dimensionality of the word vectors. Default is 100.
    epoch: Number of training epochs. Default is 5.
    min_count: Ignores words with frequency lower than this value. Default is 5.
    neg: Number of negative samples to use. Default is 5.

Fine-tuning the Model:

You can adjust the parameters for better performance depending on your specific task and dataset.

For instance:

model = fasttext.train_unsupervised('your_dataset.txt', model='skipgram', lr=0.1, dim=300, epoch=10, min_count=2)

This would train the model with a learning rate of 0.1, word vectors of size 300, 10 epochs, and ignoring words that appear less than twice.
Using Pre-trained fastText Models:

If you want to use a pre-trained fastText model instead of training one from scratch, you can download a pre-trained model from fastText's official website or other repositories and load it using:

# Loading a pre-trained model
pretrained_model = fasttext.load_model('cc.en.300.bin')  # English fastText model
word_vector = pretrained_model.get_word_vector('example')

This allows you to quickly leverage pre-trained word vectors on large corpora without the need for training from scratch.
Conclusion:

    You can train word vectors for your own dataset using fastText's unsupervised training.
    The trained model can be used to retrieve word vectors and find similar words.
    The example code demonstrates the basics, but you can further explore the parameters and customization options based on your needs.
