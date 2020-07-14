# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:04:44 2019

@author: grago
"""


import binascii
from matplotlib import pyplot as plt

# This lib convert Raw bin file to Image Array type
tmp_d = []


def convToArr(path,width,height): #Raw bin file to Image Array Type

    with open(path, "rb") as binary_file:
        #Read the whole file at once
            data = binary_file.read()
        #Return the hexadecimal representation of the binary data, byte instance
            hexa = str(binascii.hexlify(data))
            cut = hexa[2:len(hexa) - 1]

    hlist = [cut[i:i+2] for i in range(0,len(cut),2)]
    for i in range (len(hlist)):
        tmp_d.append(int(hlist[i],16))

    conv_arr = [[0 for col in range(height)] for row in range(width)]

    count = 0
    for h in range (width):
        for w in range (height):
            conv_arr[h][w] = tmp_d[count]
            count += 1
    return conv_arr


def printImage(convarr): #show Array Image

    plt.figure(figsize=(12,8))
    plt.imshow(convarr, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

