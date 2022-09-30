import torch
import struct
from fairseq import rflip_v3 as rf



def get_parity_tensor(tensor):
    '''
    Gets tensor with float to float value with parity bits
    '''
    ver=rf.get_binary(tensor)
    neg=ver<0
    copy = ver = abs(ver)
    copy = copy >> 16
    bit16 = copy%2
    copy=copy>>1
    bit15 = copy%2
    copy=copy>>1
    bit14 = copy %2
    copy = copy>>10
    bit4= copy%2
    copy = copy>>1
    bit3= copy%2
    copy = copy>>1
    bit2= copy%2
    copy = copy>>1
    bit1 = neg.int()
    # 14 15 1 16 2 3 4
    # p  p  1 p  5 6 7 
            #bit1=check2
            #bit5=check4
            #bit6=check5
            #bit7=check6
    p1=bit1^bit2^bit4
    p2=bit1^bit3^bit4
    p3=bit2^bit3^bit4
    ver=ver-bit14*2**2-bit15*2**1-bit16*1+p1*2**2+p2*2**1+p3*1        
    ver=ver+(-2)*neg.int()*ver
    

  
    return rf.binary2float(ver)
def fix(tensor):
    
    '''
    Fixes the bits but leaves the parity bits as new bits
    '''

    ver=rf.get_binary(tensor)
    neg=ver<0
    copy = ver = abs(ver)
    copy = copy >> 16
    bit16 = copy%2
    copy=copy>>1
    bit15 = copy%2
    copy=copy>>1
    bit14 = copy %2
    copy = copy>>10
    bit4= copy%2
    copy = copy>>1
    bit3= copy%2
    copy = copy>>1
    bit2= copy%2
    copy = copy>>1
    bit1 = neg.int()
    
    

    
    #bit1 p1 p2 p3 bit5 b6 b7 
    #b1   b2 b3 b4  b5  b6 b7
    #2     0  1  3   4   5  6
#    3     1  2  4   5   6  7
    #v1=int(ver[0])^int(ver[2])^int(ver[4])^int(ver[6])
    #v2=int(ver[1])^int(ver[2])^int(ver[5])^int(ver[6])
    #v3=int(ver[3])^int(ver[4])^int(ver[5])^int(ver[6])
    # 14 15 1 16 2 3 4
    # b2 b3 1 b4 5 6 7
    
    
    v1= bit14^bit1^bit2^bit4
    v2= bit15^bit1^bit3^bit4
    v3= bit16^bit2^bit3^bit4
    
    candidate=v1*1+v2*2+v3*4
    
    check5=(candidate==5)
    ver=ver+check5.int()*((-bit2+(bit2^1).int())*2**30)
    check6=(candidate==6)
    ver=ver+check6.int()*((-bit3+(bit3^1).int())*2**29)    
    check7=(candidate==7)
    ver=ver+check7.int()*((-bit4+(bit4^1).int())*2**28)            
    
    check3=(candidate==3)
    ver=ver+(check3.int())*(-2)*ver
    ver=ver+neg.int()*ver*(-2)

    return(rf.binary2float(ver))
    