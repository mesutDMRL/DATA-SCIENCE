# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:10:29 2022

@author: mesut
"""

# verileri görselleştirmek için kullandığımız kütüphane

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# plt.plot(doğrusal grafik)
# plt.bar(bar grafiği)
# plt.hist(histogram grafiği)
# plt.scatter(nokta grafiği)
# plt.stackplot(alan grafiği)
# plt.pie(pasta grafiği)

plt.plot([1,2,3,4], [1,4,9,16]) # İlk dizi x eksene. İkinci dizi y ekseni.

x = [1,2,3,4]
y = [1,4,9,16]

plt.plot(x,y) # bu şekilde kullanmakta mümkün.
plt.show()

plt.title("İlk Grafigimiz") # grafiğimize başlık verdik.
plt.xlabel("x axis") # x eksenine isim verdik.
plt.ylabel("y axis") # y eksenine başlık verdik.
plt.plot(x,y)
plt.show()

# normalde çentikler otomatik olarak yerleştirilir.
# kendimiz de çentiklerin yerini tanımlayabiliriz.

plt.xticks([1,2,3,4]) # küsüratları da gösterebilir.
plt.yticks([1,4,9,16])
plt.title("İlk Grafiğimiz!")
plt.xlabel("x axis") 
plt.ylabel("y axis")
plt.plot(x, y)
plt.show()

# LEJANT KOYMA

plt.plot(x,y, label = "x^2", color = "purple")
plt.xticks([1,2,3,4,5]) 
plt.yticks([1,4,9,16,25])
plt.title("İlk Grafiğimiz!")
plt.xlabel("x axis") 
plt.ylabel("y axis")
plt.legend() # bunu kullanmazsak lejant'ı göstermez. lejant: doğrunun türünü gösteren sağ üstteki kutucuk
plt.show()

# LİNEWİDTH(DOĞRU GENİŞLİĞİ) EKLEME

plt.plot(x,y, label = "x^2", color = "purple", linewidth = 1, linestyle = "--" , marker = ".")
plt.xticks([1,2,3,4,5]) 
plt.yticks([1,4,9,16,25])
plt.title("İlk Grafiğimiz!")
plt.xlabel("x axis") 
plt.ylabel("y axis")
plt.legend()
plt.show()

# AYNI GRAFİKTE BİRDEN FAZLA ÇİZİM YAPMAK

plt.plot(x,y, label = "x^2", color = "purple", linewidth = 1, linestyle = "--" , marker = ".")
plt.xticks([0,1,2,3,4,5]) 
plt.yticks([0,1,4,9,16,25])
plt.title("İlk Grafiğimiz!")
plt.xlabel("x axis") 
plt.ylabel("y axis")

# 2. çizimi yapıyoruz.
x2 = np.arange(0,5,0.5) # 0 ile 5 arasında 0.5 arttırarak bir dizi oluşturur.
plt.plot(x2, x2*2, color = "red", linewidth = 2, marker=".", label="2*x")

plt.legend()
plt.show()

# GRAFİĞİ BİLGİSAYARA KAYDETME

plt.plot(x,y, label = "x^2", color = "purple", linewidth = 1, linestyle = "--" , marker = ".")
plt.xticks([0,1,2,3,4,5]) 
plt.yticks([0,1,4,9,16,25])
plt.title("İlk Grafiğimiz!")
plt.xlabel("x axis") 
plt.ylabel("y axis")
# 2. çizimi yapıyoruz.
x2 = np.arange(0,5,0.5) 
plt.plot(x2, x2*2, color = "red", linewidth = 2, marker=".", label="2*x")
plt.legend()

plt.savefig("İlk_Grafiğim.png", dpi=300) # aynı dosya yolu üzerine kaydeder. dpi'ı artırıp daha ayrıntılı grafikler elde edebiliriz.
# plt.show() dan önce kullanmamız gerekiyor.
plt.show()

# BARCHART

x = ["Ankara", "İstanbul", "İzmir"] 
y = [120, 178, 87]

plt.bar(x, y) # grafiği barlarla ifade etmek için kullanılır.
plt.show()

# Eğer çubuklardan bir tanesini diğerlerinden farklılaştırmak için içini boyamak istersek;

x = ["Ankara", "İstanbul", "İzmir"]
y = [120, 178, 87]

cubuklar = plt.bar(x,y)
cubuklar[1].set_hatch("/") # birinci index'i yani istanbul'u / şeklinde ifade ettik.
cubuklar[0].set_hatch("*") # sıfırıncı index'i yani ankara'yı * şeklinde ifade ettik.
cubuklar[2].set_hatch(".") 
plt.show()

# DETAYLI ÖRNEKLER

gas = pd.read_csv("petrolfiyatlari.csv")
plt.title("Petrol Fiyatları")
plt.plot(gas["Year"], gas["USA"], "b-", label="USA") # "b-" bu ifade de b harfi blue demektir. - ise - şeklinde çiz demektir.

plt.xlabel("Yıl")
plt.ylabel("Dolar")

plt.legend()
plt.show()


plt.title("Petrol Fiyatları")
plt.plot(gas.Year, gas.USA, "b-", label="USA") # arada boşluk yoksa köşeli parantez kullanmadan da sütunları kullanabilirim.
plt.plot(gas.Year, gas["Canada"], "r-", label="Canada")
plt.plot(gas.Year, gas["South Korea"], "g.-", label="South Korea")
plt.plot(gas.Year, gas["France"], "y.-", label="France")
plt.plot(gas["Year"],gas.Germany, color = "orange", label="Germany")
plt.xlabel("Yıl")
plt.ylabel("Dolar")

plt.legend()
plt.show()

# FİGURE SİZE(TABLO BÜYÜKLÜĞÜ)

plt.figure(figsize = (9,4)) # x'in uzunluğunu 9 y'nin uzunluğunu 4 verdik.
plt.title("Petrol Fiyatları")
plt.plot(gas.Year, gas.USA, "b-", label="USA") 
plt.plot(gas.Year, gas["Canada"], "r-", label="Canada")
plt.plot(gas.Year, gas["South Korea"], "g.-", label="South Korea")
plt.plot(gas.Year, gas["France"], "y.-", label="France")
plt.plot(gas["Year"],gas.Germany, color = "orange", label="Germany")
plt.xlabel("Yıl")
plt.ylabel("Dolar")

plt.legend()
plt.savefig("İkinci_Grafiğim.png", dpi=300)
plt.show()


