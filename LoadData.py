#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:09:35 2018

@author: LuoJiacheng
"""
import numpy as np
import pandas as pd


class LoadData(object):
    def __init__(self,ratio):
        file = './datasets/ratings.txt'
        f_file = './datasets/trust.txt'
        data = np.loadtxt(file)
        fdata = np.loadtxt(f_file).astype(int)

        data[:,2] = data[:,2]/5
        indices = np.arange(len(data[:,2]))
        num_train_samples = int(len(indices)*ratio)
        
        np.random.seed(615)
        np.random.shuffle(indices)
        
        user = data[:,0].copy().astype(int)
        item = data[:,1].copy().astype(int)
        rating = data[:,2].copy()
        self.user = user[indices]
        self.item = item[indices]
        self.rating = rating[indices]
        
        self.train_user = self.user[:num_train_samples]
        self.train_item = self.item[:num_train_samples]
        self.train_rating = self.rating[:num_train_samples]
        
                
        self.test_user = self.user[num_train_samples:]
        self.test_item = self.item[num_train_samples:]
        self.test_rating = self.rating[num_train_samples:]
        #find friends
        
        friends_dic = {}
        for elem in fdata[:,0]:
            friends_dic.setdefault(elem,[])
        for i,elem in enumerate(fdata[:,0]):
            friends_dic[elem].append(fdata[i,1])
        self.friends_dic = friends_dic
        
        friends = []
        for usr in self.user:
            try:
                friends.append(self.friends_dic[usr][0])
            except:
                friends.append(0)
        friends = np.array(friends)
        self.friends = friends
        
        self.train_friends =self.friends[:num_train_samples]
        self.test_friends = self.friends[num_train_samples:]
        
        #sim matrix
        matrix = np.zeros([max(fdata[:,1]),max(self.item)])
        matrix[self.train_user,self.train_item]=self.train_rating
        self.matrix = matrix        
        
        #sim
        self.train_sim = self.comput_sim(self.train_friends,self.train_user)
        
        self.test_sim =  self.comput_sim(self.test_friends,self.test_user)

    def comput_sim(self,user,friends):
        sim = []
        for i,j in zip(user,friends):
            sim.append(self.sim_pearson(self.matrix[i,:],self.matrix[j,:]))
        sim = ((np.array(sim)+1)/2)
        return sim
        
    def sim_pearson(self,user1,user2):
#    a=np.isfinite(user1)
#    b=np.isfinite(user2)
        a_=np.nan_to_num(user1)
        b_=np.nan_to_num(user2)
        user1[user1 == 0] = np.nan
        user2[user2 == 0] = np.nan
        a=a_>0
        b=b_>0
        n=0
        k = np.logical_and(a,b)
        n = k[k==True].size
        if n==0:
            return 0
        user1_k = user1[k]
        user2_k = user2[k]
        num = np.sum((user1_k-np.nanmean(user1))*(user2_k-np.nanmean(user2)))
        den = np.sqrt(np.sum(pow((user1_k-np.nanmean(user1)),2))*np.sum(pow((user2_k-np.nanmean(user2)),2)))
        if den ==0 : return 0
        sim = num/den
        sim_ac = sim*(2*user1_k.size)/(user1[pd.notnull(user1)].size+user2[pd.notnull(user2)].size)
        return sim_ac
    
    def get_batches(self,user,item, Y,batch_size):
        '''
        数据batch迭代器
        user_batch, item_batch,y_batch
        
        return 1 friends ,you can revise
        '''
        
        if batch_size is None:
            batch_size = 128
        
        n_batches = int(len(Y) / batch_size)
        # 这里我们仅保留完整的batch，对于不能整出的部分进行舍弃
        
        while (True):
            # inputs
            
            for count in range(n_batches):
                friends =[]
                
                user_batch = user[count*batch_size:(count+1)*batch_size]
                item_batch = item[count*batch_size:(count+1)*batch_size]
                
                # targets
                y_batch = Y[count*batch_size:(count+1)*batch_size]
                
                for usr in user_batch:
                    try:
                        friends.append(self.friends_dic[usr][0])
                    except:
                        friends.append(0)
                friends = np.array(friends)

                sim = self.comput_sim(user_batch,friends)
                sim = np.reshape(sim,(batch_size,1))
                
                user_batch = np.reshape(user_batch,(batch_size,1))
                item_batch =  np.reshape(item_batch,(batch_size,1))
                friends =  np.reshape(friends,(batch_size,1))
                y_batch = np.reshape(y_batch,(batch_size,1))
                yield user_batch, item_batch,friends,sim,y_batch