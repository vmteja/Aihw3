Purpose: To build and train a multi-layer neural network in TensorFlow.

 For this homework, I want you to build and train a simple multi-layer neural network in TensorFlow. Your neural network will operate on strings of length 40 over the alphabet {A,B,C,D}. Since it is close to Halloween, we can imagine these letters are coming from chemicals in some alien's genetic code (or if you prefer vampires, you can think of this in terms of the vampire genetics from the Blood+ anime). The goal is that it should be able to classify strings as NONSTICK, 12-STICKY, 34-STICKY, 56-STICKY, 78-STICKY, STICK_PALINDROME.

To understand what these classes mean, let's first define stickiness. The letter A sticks with the letter C (and vice-versa) and the letter B sticks with the letter D (and vice-versa). Given two strings u and v, u sticks with v if len(u)=len(v) and for all i<len(u), the letter u[i] sticks with v[i]. For example, AABDC sticks with CCDBA. As an alien geneticist will tell you, if an alien chromosome has a lot of regions which are sticky, it can help protect the chromosome from mutations. This might be especially important for alien sexual chromosomes . GENECo has developed a tool which splits alien genetic info into strings which are precisely 40 character long. Given a string w let wR denote the string written backwards (in reverse). A 40 character string is a stick palindrome if it can be written as the concatenations of two strings vw and v sticks with wR. A 40 character string is k-sticky, if it can be written as a concatenation of three string uvw such that len(u)=k and u sticks with wR. After training, your network should output NONSTICK on any string which is not even 1-sticky; otherwise, it should output k(k+1)-STICKY if it is either k-sticky or k+1-sticky, and it is not also in the next class in our list of classes above. Finally, it should output STICK_PALINDROME only if the string is a stick palindrome. 

As an example, you might run gene_snippet_generator.py with concrete with the values:

python sticky_snippet_generator.py 3 .1 3 test_data.txt

This might write to the file test_data.txt the lines:

ABCBDCBDBCADBACBACDADACBDADBACBDACBBADC
BACBDCBDBBACBACDADDADBDACBBACACBDACBDCD
DDAADCBDBBADACACBDBDACBBABDACBBADACBCBB

Given the above data sets I want you to design and conduct experiments which follow the guidelines for experiments from the Oct 11 Lecture and which answer the following questions:

1) What is the effect of training set size on how good the trained model is? How does this compare to the number of weights that your model has?

A)

Hypothesis: Increase in training size increases the accuracy.
Description: We trained the model with different train_folders and recorded the following results
Results:
Training size	Accuracy
5000	71.6%
10000	72.2%
20000	75.7%
60000	75.3%

Conclusion: The hypothesis is true.


2) What is the difference between training on data chosen completely at random versus on well chosen examples? 

A)
Hypothesis 1: When the distribution of classes in training data is equally chosen, the model performs well.
Hypothesis 2: Class distribution of testing data should not matter. 

Description: 
a)	We experimented by randomly on training sets which are of equal class distribution as well as choosing only one class (for e.g class 12 sticky being 70% of the training set).
b)	Randomly generated test data v/s test data from only one class.

Conclusion:

a)	For effective training of the model, the class distribution has to be nearly similar as then the model will be equally equipped to predict all the different classes.

b)	It doesn’t make a difference as in the end the model has to predict whatever data it has. 

3) What is the difference in accuracy in using cross-validation for testing versus using separate test data?

A)
Accuracy in using cross-validation for testing versus using separate test data.
Testing	Accuracy
Cross-validation	80.3% for 10,000 training size
Separate test data	72% for 10,000 testing

Conclusion
As we have run this exp with different combinations of test and train on the same set of data 5 times (k=5) cross validation score is higher, as it is taking average of those scores. It has seen all and been verified the possible combinations of training data. 
When you have separate test data, you are really checking against one class at a time, hence it is lower.

