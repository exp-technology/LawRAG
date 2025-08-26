with open("data2.txt", "r") as file:
    isi = file.read()
    print(isi)
print('a')
# w -> write
# r -> read
# file = open('data2.txt', 'r')

# BMI = weight (kg) / [height (m)]²
# BMI Categories:
# Underweight: BMI less than 18.5, Normal weight: BMI between 18.5 and 24.9, Overweight: BMI between 25 and 29.9, and Obese: BMI of 30 or greater.

# bermain dengan input
berat = input("masukan berat badan mu [kg]: ")
tinggi = input('masukan tinggi badan mu [m]: ')
bmi = float(berat) / float(tinggi) ** 2
print("bmi index mu adalah : ", bmi)
if bmi < 18.5:
    print("you are underweight")
elif bmi >= 18.5 and bmi < 25:
    print("your bmi is normal")
elif bmi >= 25 and bmi < 30:
    print("you are overweight")
else:
    print("you are obese")


# bermain dengan file luar, dan kumpulkan angka #############

# buat 1000 angka
import random
# menggunakan library random
data1000 = []
for g in range(1000):
    random_angka = random.randint(0, 100)
    data1000.append(random_angka)

with open('data2.txt', 'w') as tulis:
    for g in data1000:
        tulis.write(str(g) + '\n')

angka0_100 = list(range(101))

with open('data2.txt', 'r') as baca:
    hasil_baca = baca.read()
hasil_baca = hasil_baca.split('\n')
hasil_baca = hasil_baca[0:-1]

tampung_jumlah = []
for g in angka0_100:
    hitungan_sementara = 0
    for j in hasil_baca:
        if g == int(j):
            hitungan_sementara = hitungan_sementara + 1
    tampung_jumlah.append([g, hitungan_sementara])


with open('data2_agregat.txt', 'w') as tulis:
    for j in tampung_jumlah:
        tulis.write("data "+ str(j[0]) + " : " + str(j[1]))

