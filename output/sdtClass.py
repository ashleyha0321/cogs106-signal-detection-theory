import numpy as np
import scipy
import matplotlib.pyplot as plt


# class was created using the assistance of chatGPT


class SignalDetection:
   def __init__(hits, misses, falseAlarms, correctRejections):
       self.hits = hits
       self.misses = misses
       self.falseAlarms = falseAlarms
       self.correctRejections = correctRejections


   def hit_rate(self):
       return (self.hits + 0.5) / (self.hits + self.misses + 1)
  
   def false_alarm_rate(self):
       return (self.falseAlarms + 0.5) / (self.falseAlarms + self.correctRejections + 1)
  
   def d_prime(self):
       hit_rate = self.hit_rate()
       false_alarm_rate = self.false_alarm_rate()
       return np.round(scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(false_alarm_rate), 6)
  
   def criterion(self):
       hit_rate = self.hit_rate()
       false_alarm_rate = self.false_alarm_rate()
       return np.round(-0.5 * (scipy.stats.norm.ppf(hit_rate) + scipy.stats.norm.ppf(false_alarm_rate)), 6)
  
   def compute(self):
       return self.d_prime(), self.criterion()


