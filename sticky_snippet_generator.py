

import numpy as np
import sys
from label_classifier import get_label


def mutate_def(listname):
    l = len(listname)
    mutation_rate_by_3 = 1/3
    mutation_rate_rest = 1 - (2*mutation_rate_by_3)
    for i in range(0, l):
        if (listname[i] == 'A'):
            ch = np.random.choice(
                ['A', 'B', 'C', 'D'],
                1,
                p=[0.0, 0.333,0.333,0.334])
            listname.append(ch[0])
        elif (listname[i] == 'B'):
            ch = np.random.choice(
                ['B', 'A', 'C', 'D'],
                1,
                p=[0.0, 0.333,0.333,0.334])
            listname.append(ch[0])
        elif (listname[i] == 'C'):
            ch = np.random.choice(
                ['C', 'B', 'A', 'D'],
                1,
                p=[0.0, 0.333,0.333,0.334])
            listname.append(ch[0])
        else:
            ch = np.random.choice(
                ['D', 'A', 'B', 'C'],
                1,
                p=[0.0, 0.333,0.333,0.334])
            listname.append(ch[0])

    return listname[l:]



def mutate(listname, mutation_rate):
    mutation_rate_by_3 = mutation_rate / 3
    mutation_rate_rest = 1 - mutation_rate_by_3 * 3

    l = len(listname)
    for i in range(0,l):
        if(listname[i]=='A'):
            ch = np.random.choice(
          ['A','B', 'C', 'D'], 
          1,
          p = [mutation_rate_rest, mutation_rate_by_3, mutation_rate_by_3, mutation_rate_by_3])
            listname.append(ch[0])
        elif(listname[i]=='B'):
            ch = np.random.choice(
          ['B','A', 'C', 'D'], 
          1,
          p = [mutation_rate_rest, mutation_rate_by_3, mutation_rate_by_3, mutation_rate_by_3])
            listname.append(ch[0])
        elif(listname[i]=='C'):
            ch = np.random.choice(
          ['C','B', 'A', 'D'], 
          1,
          p =[mutation_rate_rest, mutation_rate_by_3, mutation_rate_by_3, mutation_rate_by_3] )
            listname.append(ch[0])
        else:
            ch = np.random.choice(
          ['D','A','B', 'C'], 
          1,
          p = [mutation_rate_rest, mutation_rate_by_3, mutation_rate_by_3, mutation_rate_by_3])
            listname.append(ch[0])
    
    
        
    return listname[l:]


# In[85]:

def generate_sticky():
    v = np.random.choice(
      ['A', 'B', 'C', 'D'], 
      20,
      p=[0.25, 0.25, 0.25, 0.25])

    w=[]
    for x in range(0,len(v)):
        if(v[x] =='A'):
            w.append('C')
        elif(v[x] =='C'):
            w.append('A')
        elif(v[x] =='B'):
            w.append('D')
        elif(v[x] =='D'):
            w.append('B')
        

    v = np.ndarray.tolist(v)
    w=w[::-1] #w reverse
    s = v+w
    return s
    


# In[98]:


if __name__ == "__main__":
    s=""
    num_snippets = int(sys.argv[1])
    mutation_rate = float(sys.argv[2])
    from_ends = int(sys.argv[3])
    output_file = sys.argv[4]
    pfile = open(output_file,"w+")
    pfile.seek(0)
    pfile.truncate()
    initial = generate_sticky()
    ini2 = generate_sticky()
    ini3 = generate_sticky()
    n=0
    
    while(n<num_snippets):
        starts = initial[:from_ends]
        middle=initial[from_ends:40-from_ends]
        ends = initial[40-from_ends:]

        starts = mutate(starts,mutation_rate)
        ends = mutate(ends,mutation_rate)
        middle = mutate_def(middle)
    
        for x in range(0,from_ends):
            middle.insert(x,starts[x])
        for x in range(0,from_ends):
            middle.append(ends[x])
        output = open(output_file, "a+")
        for char in middle:
            output.write(char)
        output.write("\n")
        s = "".join(middle)
        print get_label(s)
        n+=1
        initial = middle
        output.close()
    
    

