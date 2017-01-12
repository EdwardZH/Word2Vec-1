# Word2Vec
A word2vec algorithm implemented in TensorFlow

The rise of TensorFlow over the past year has been amazing. It is now one of the most popular open source projects on GitHub and certainly the fastest growing deep learning library available. At the time of writing, it has amassed more GitHub stars than Linux, with 42,769 and 40,828 respectively.

It is also incredibly portable, running on a multitude of platforms, ranging from Raspberry Pi, Android and Apple mobile devices through to 64-bit desktop and server systems. Furthermore, in May 2016, Google announced the creation of its tensor processing unit or TPU, which is a custom ASIC built specifically for machine learning and tailored for TensorFlow, which now operate in its data centres. So long-term investment and support is there.

The other superstar in the machine-learning world is the word2vec algorithm released by Tomas Mikolov and a team from Google in January 2013. This was based on their paper, “Efficient Estimation of Word Representations in Vector Space”. I have written before about the incredible properties of word embeddings created by this algorithm.

Word2vec and TensorFlow seem like a perfect match, both emerging from Google, the machine-learning version of a super couple. However, the few implementations I have seen so far have been disappointing, so I decided to write my own.

Some of the key things I wanted to achieve were:

- a robust method to clean and tokenise text
- the ability to process very large text files
- make full use of TensorFlow GPU support for fast training
- use TensorFlow FIFO queues to eliminate I/O latency
- simple code that could be used by people learning TensorFlow
- a way to demonstrate the model once trained

I am very pleased with result. Even with a basic Nvidia GTX 750 Ti we can process an entire Wikipedia training epoch in less than 4 hours.

Follow me on Twitter for more updates like this.

@robosoup www.robosoup.com
