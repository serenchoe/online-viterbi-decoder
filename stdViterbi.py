# Standard Viterbi Based on A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition
# Online Viterbi based on The on-line Viterbi algorithm (Master’s Thesis) by Rastislav Šrámek

from pyllist import dllist, dllistnode
import random
import numpy as np
import time

K = 4             # # of Hidden States
M = 4             # # of Observation Symbols
T = 10           # # of time instances

A = [ [ 0.96, 0.04, 0.0, 0.0 ],              # Transition Probability Matrix
      [ 0, 0.95, 0.05, 0.0 ], 
      [ 0.0, 0.0, 0.85, 0.15 ], 
      [ 0.1, 0.0, 0.0, 0.9 ] ]

E = [ [ 0.6, 0.2, 0.0, 0.2 ],                  # Emission Matrix
      [ 0.1, 0.8, 0.1, 0.0 ],  
      [ 0.0, 0.14, 0.76, 0.1 ],    
      [ 0.1, 0.0, 0.1, 0.8 ] ] 

initial =  [ 0.25, 0.25, 0.25, 0.25 ]            # Initial distribution


### /////////////////////////// Online viterbi data structures ////////////////////



global prob_list
global state_list
global node_list

prob_list = dllist()             # probability list                              [ prob[stete_size] ]
state_list = dllist()            # state list                                    [ state[stete_size] ]
node_list = dllist()             # linked list for survivor memory               [ state, time, parent, num_children]

global root
root = None

global prev_root
prev_root = None

global delta_t
delta_t = None

global decoded_stream
decoded_stream = []


def clear_dllist(dl_list):
   llist = dl_list.last
   while(llist!=None):
      temp = llist.prev
      dl_list.remove(llist)
      llist = temp



def clear_all_lists():
   clear_dllist(prob_list)
   clear_dllist(state_list)
   clear_dllist(node_list)
 


def online_viterbi_initialization(starting_state):
   global root
   global prev_root
   global decoded_stream

   root = None
   prev_root = None
   decoded_stream.clear()
   clear_all_lists()

   initial_prob = []
   initial_state = []

   for i in range(K):
      initial_prob.append(initial[i])
      initial_state.append(starting_state)
   prob_list.append(initial_prob)
   state_list.append(initial_state)





def compress(current_time):

   current = node_list.last
   while (current!= None):
      state = current.value[0]
      time = current.value[1]
      parent = current.value[2]
      num_children = current.value[3]
      temp = None

      if (num_children==0 and time!=current_time):
         if (parent!=None):
            parent.value[3] = parent.value[3] - 1 
      else:
         while(parent!=None and parent.value[3]==1):
            # parent.value[3] = -1
            current.value[2] = current.value[2].value[2]
            parent = current.value[2]

      current = current.prev


def free_dummy_nodes(current_time):
   # Free remaininig nodes with 0 children that are not leaves
   current = node_list.last
   while (current!= None):
      state = current.value[0]
      time = current.value[1]
      parent = current.value[2]
      num_children = current.value[3]
      temp = current.prev

      if (num_children<=0 and time!=current_time):
         node_list.remove(current)

      current = temp







def find_new_root():
   global root
   global prev_root
   global delta_t
   # //Returns true if root has changed based on time delta between previous root and new root


   # first make sure path has merged
   if (root==None):
      last = node_list.last
      traced_root = [None] * K
      leaf = last
      for i in range(K):
         current = leaf
         while (current != None) : 
            temp = current
            current = current.value[2]
            if (current==None):
               traced_root[i] = temp
         leaf = leaf.prev

      result = False
      if len(traced_root) > 0 :
          result = all(elem == traced_root[0] for elem in traced_root)

      if result==False:
         return False



   # find new root
   current = node_list.last
   # aux = node_list.last
   aux = None
   time = current.value[1]

   delta_t = current.value[1]

  #  /*
  #  Find last node that has at least 2 children starting from any leave or node with at least two children

  #  Proof and analysis of this algorithm can be found on 
  #  "Šrámek R., Brejová B., Vinař T. (2007) On-Line Viterbi Algorithm for Analysis of Long Biological Sequences. 
  #  In: Giancarlo R., Hannenhalli S. (eds) Algorithms in Bioinformatics. WABI 2007. 
  #  Lecture Notes in Computer Science, vol 4645. Springer, Berlin, Heidelberg"
  # */
  
   while (current != None) : 
      if (current.value[3] >= 2):
         aux = current;

      current = current.value[2]

   if (aux != None):
      if (root==None):
         root = aux
         delta_t = delta_t - aux.value[1]
         if(delta_t == 0):
            return False
         else: 
            # print("New root : {} ".format(root.value[1]))            
            return True
      else:
         if(aux != root):  # Test if root has changed
            delta_t = delta_t - aux.value[1]
            if(delta_t == 0):
               return False
            else:
               prev_root = root
               root = aux

               # print("Prev root : {} ".format(prev_root.value[1]))
               # print("New root : {} ".format(root.value[1]))

               return True
   else:
      return False








def traceback():
   global root
   global prev_root
   global delta_t
   global decoded_stream

   interim_decoded_stream = []
   p_col = prob_list.last
   s_col = state_list.last

   output = root.value[0]    # state
   print("{} ".format(output), end='')
   interim_decoded_stream.append(output)

   for i in range(delta_t):
       if(s_col != None):    # Find column corresponding to root
         s_col = s_col.prev

       if(p_col != None):    # Find column corresponding to root
         p_col = p_col.prev;         

   if (prev_root==None):
      depth = root.value[1]
   else:
      depth = (root.value[1] - prev_root.value[1] - 1)

   for i in range(depth):
      output = s_col.value[output]
      print("{} ".format(output), end='')
      interim_decoded_stream.append(output)      
      S = s_col;
      s_col = s_col.prev
      state_list.remove(S)
      P = p_col
      p_col = p_col.prev
      prob_list.remove(P)


   # remove further remaining 
   while (p_col!=None):
      S = s_col;
      s_col = s_col.prev
      state_list.remove(S)
      P = p_col
      p_col = p_col.prev
      prob_list.remove(P)

   print("")

   interim_decoded_stream.reverse()
   decoded_stream.extend(interim_decoded_stream)





def traceback_last_part():
   global root
   global decoded_stream

   interim_decoded_stream = []
   p_col = prob_list.last
   s_col = state_list.last

   output = p_col.value.index(max(p_col.value))
   print("{} ".format(output), end='')
   interim_decoded_stream.append(output)

   if (root==None):
      depth = (T-1)
   else:
      depth = (T-1) - root.value[1] - 1

   for i in range(depth):
      output = s_col.value[output]
      print("{} ".format(output), end='')
      interim_decoded_stream.append(output)
      S = s_col;
      s_col = s_col.prev
      # state_list.remove(S)
      P = p_col
      p_col = p_col.prev
      # prob_list.remove(P)

   print("")

   interim_decoded_stream.reverse()
   decoded_stream.extend(interim_decoded_stream)







def update(t, observation):
   # /*
   #    Online Viterbi algorithm : Updates scores and paths (max_idexes) matrices when the observation at time t in received
   #    observation: observation at time t
   # */

   p_col = prob_list.last
   s_col = state_list.last
   last_node = node_list.last

   pCol = [0] * K
   sCol = [0] * K

   # Create new matrices columns

   for j in range(K):
      max = -1
      aux = -1
      max_index = 0

      for i in range(K):
         aux = p_col.value[i]*A[i][j]*E[j][observation]
         if (aux > max):
            max = aux
            max_index = i      

      # Store score and path
      pCol[j] = max
      sCol[j] = max_index

      # add node
      if (t==0):
         parent_node = None
      else:
         temp = K - max_index - 1
         parent_node = last_node
         while (temp > 0):
            parent_node = parent_node.prev
            temp = temp - 1
         parent_node.value[3] = parent_node.value[3] + 1

      node_list.append( [j, t, parent_node, 0] )

   prob_list.append(pCol)
   state_list.append(sCol)

   # print("Before compress")
   # printList()
   # printProbList()
   # printStateList()

   compress(t)
   free_dummy_nodes(t)

   if(find_new_root()):
      traceback()

   # print("After compress")
   # printList()





def printProbList():
   prob = prob_list.last
   while(prob != None):
      print(prob)
      prob = prob.prev

   print("\n\n")




def printStateList():
   state = state_list.last
   while(state != None):
      print(state)
      state = state.prev

   print("\n\n")








def printList():
   node = node_list.last
   while(node != None):
      print(node)
      node = node.prev

   print("\n\n")












# ///////////////////////////////////////////////////////////////////

scores = [ [0] * T  for _ in range(K) ] 
path = [ [0] * T  for _ in range(K) ] 
optimalPath = [0] * T

def std_viterbi_initialization(observations):
   # for i in range(K):
   #    scores[i][0] = initial[i]*E[i][observations[0]]

   for j in range(K):
      max = -1
      aux = -1
      max_index = 0

      for i in range(K):
         aux = initial[i]*A[i][j]*E[j][observations[0]];
         if (aux > max):
            max = aux
            max_index = i

      scores[j][0] = max
      path[j][0] = max_index





def std_viterbi_recursion(observations):
   max = -1
   aux = -1
   max_index = 0
   for t in range(1, T):
      for j in range(K):
         max = -1
         aux = -1
         max_index = 0

         for i in range(K):
            aux = scores[i][t-1]*A[i][j]*E[j][observations[t]];
            if (aux > max):
               max = aux
               max_index = i

         scores[j][t] = max
         path[j][t] = max_index







def std_viterbi_termination():
   max_index = 0
   max = 0
   for j in range(K):
      if(scores[j][T-1] > max):
         max = scores[j][T-1]
         max_index = j

   optimalPath[T-1] = max_index

   for t in range(T-2, -1, -1):
      optimalPath[t] = path[optimalPath[t + 1]][t + 1]





def std_viterbi(observations):
   std_viterbi_initialization(observations)
   std_viterbi_recursion(observations)
   std_viterbi_termination()








def printArray(array):
   for j in range(len(array)) :
        print("{} ".format(array[j]), end='')






# if __name__ == '__main__':
#    # observations = [3, 2, 0, 3, 1, 3, 3, 1, 2, 2]
#    # observations = [ 1, 1, 1, 1, 1, 1, 1, 2, 2, 2 ]
#    # observations = [ 1, 3, 3, 2, 0, 2, 0, 2, 2, 0 ]
#    # observations = [ 1, 2, 3, 1, 3, 2, 3, 2, 1, 3 ]
#    observations = [ 2, 0, 3, 2, 3, 2, 3, 0, 1, 0 ]
#    # observations = [0] * T

#    # observed_states =states_list = np.array([1, 2, 3, 8, 12, 17, 24, 25, 30, 25, 26, 27, 31], dtype=int) 
#    # observations = (observed_states - 1).tolist()

#    previous = 0

#    online_viterbi_initialization(0)

#    for i in range(T):
     

#       # observations[i] = int( ( previous + (2 * random.random())%2 ) % K )
#       # previous = observations[i]
      
#       update(i, observations[i])



#    traceback_last_part()

#    print("\nobservations:  ", end='')
#    printArray(observations)
#    std_viterbi(observations)
#    print("\nStd Viterbi window:  ", end='')
#    printArray(optimalPath)
#    print("\nOnline Viterbi window:  ", end='')
#    printArray(decoded_stream)

#    print("\n\n")




if __name__ == '__main__':
   # observations = [3, 2, 0, 3, 1, 3, 3, 1, 2, 2]
   # observations = [ 1, 1, 1, 1, 1, 1, 1, 2, 2, 2 ]
   # observations = [ 1, 3, 3, 2, 0, 2, 0, 2, 2, 0 ]
   # observations = [ 1, 2, 3, 1, 3, 2, 3, 2, 1, 3 ]
   # observations = [ 2, 0, 3, 2, 3, 2, 3, 0, 1, 0 ]
   observations = [0] * T

   # observed_states =states_list = np.array([1, 2, 3, 8, 12, 17, 24, 25, 30, 25, 26, 27, 31], dtype=int) 
   # observations = (observed_states - 1).tolist()

   previous = 0

   online_viterbi_initialization(0)

   for i in range(100*60*60):
     
      count = i%T
      observations[count] = int( ( previous + (2 * random.random())%2 ) % K )
      previous = observations[count]
      
      update(count, observations[count])

      if(count == T-1):
         traceback_last_part()

         print("\nobservations:  ", end='')
         printArray(observations)
         std_viterbi(observations)
         print("\nStd Viterbi window:  ", end='')
         printArray(optimalPath)
         print("\nOnline Viterbi window:  ", end='')
         printArray(decoded_stream)

         print("\n\n")

         # for fresh new start of online viterbi decoding
         online_viterbi_initialization(0);

      time.sleep(0.1)

