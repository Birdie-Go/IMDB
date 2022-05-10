import matplotlib.pyplot as plt
import numpy

fp1 = open('word2vec_loss.txt', 'r')
ls1 = []
for line in fp1:
    line = line.strip('\n')
    if ':' not in line:
        continue
    line = line.split(':')[1]
    ls1.append(float(line.split(' ')[2]))
ls1 = numpy.array(ls1, dtype = float)
print(ls1)
fp1.close()

fp2 = open('tf_loss.txt', 'r')
ls2 = []
for line in fp2:
    line = line.strip('\n')
    if ':' not in line:
        continue
    line = line.split(':')[1]
    ls2.append(float(line.split(' ')[2]))
ls2 = numpy.array(ls2, dtype = float)
fp2.close()

fp3 = open('tfidf_loss.txt', 'r')
ls3 = []
for line in fp3:
    line = line.strip('\n')
    if ':' not in line:
        continue
    line = line.split(':')[1]
    ls3.append(float(line.split(' ')[2]))
ls3 = numpy.array(ls3, dtype = float)
fp3.close()


input_values = numpy.arange(0, len(ls1), 1, dtype = int)
plt.plot(input_values, ls1)
plt.plot(input_values, ls2)
plt.plot(input_values, ls3)
plt.title("loss function", fontsize = 24)
plt.xlabel("epoch", fontsize = 14)
plt.ylabel("loss", fontsize = 14)
plt.legend(['word2vec', 'tf', 'tf-idf'], loc = 'upper left')
# plt.tick_params(axis='both', labelsize = 14)
plt.show()