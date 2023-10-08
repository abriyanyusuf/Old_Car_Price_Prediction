 # Laporan Proyek Machine Learning - Abriyan Yusuf

## Domain Proyek

Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

Mobil tua merupakan salah satu alternatif sebagian orang yang ingin memiliki kendaraan dengan biaya yang terbatas. Hal tersebut menjadikannya sebagai kendaraan yang banyak dicari oleh sebagian orang tak terkecuali pengusaha jual beli mobil. Untuk meraih keuntungan yang baik pengusaha mobil tua memerlukan suatu cara untuk menilai harga dari setiap mobil tua yang ingin dibelinya dan dijual kembali. Ketika seorang pengusaha mampu mengetahui perkiraan harga mobil tua secara akurat maka ia dapat melakukan tawar menawar sedemikian rupa dengan penjual mobil tua sehingga pengusaha tersebut dapat mendapatkan keuntungan ketika menjualnya kembali ke konsumen.


## 1. Business Understanding
### 1.1 Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi harga mobil tua untuk menjawab permasalahan berikut :
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga mobil tua?
- Berapa harga pasar mobil tua dengan karakteristik atau fitur tertentu?

### 1.2 Goals
Untuk menjawab pertanyaan tersebut, kita akan membuat predictive modelling dengan tujuan atau goals sebagai berikut :
- Mengetahui fitur yang paling berkorelasi dengan harga mobil tua
- Membuat model machine learning yang dapat memprediksi harga mobil tua seakurat mungkin berdasarkan fitur-fitur yang ada

#### 1.3 Metodologi
Prediksi harga adalah tujuan yang ingin kita capai. Harga merupakan variabel kontinu. Dalam predictive analytics, ketika membuat prediksi variabel kontinu maka kita sedang menyelesaikan **permasalahan regresi**. Oleh karena itu metodologi dalam proyek ini adalah membangun model regresi dengan harga mobil tua sebagai target.


#### 1.4 Metrik
Metrik digunakan untuk mengevaluasi seberapa baik model yang kita buat dalam memoprediksi harga. Untuk kasus regresi, beberapa metrik yang umum dan bisa digunakan diantaranya:
- Mean Squared Error (MSE), atau
- Root Mean Square Error (RMSE)
Secara umum kedua metrik tersebut mengukur seberapa jauh hasil prediksi dengan nilai sebenarnya. 

Pengembangan model juga akan menggunakan beberapa algoritma machine learning diantaranya
- K-Nearest Neighbor
- Random Forest
- Boosting Algorithm
Dari ketiga model tersebut akan dipilih satu model yang memiliki **nilai kesalahan prediksi terkecil**. Dengan kata lain kita akan membuat model seakurat mungkin yang model dengan kesalahan sekecil mungkin.


## 2. Data Understanding
Data yang akan digunakan dalam proyek ini bersumber dari Kaggle berjudul  [Old Car price Prediction](https://www.kaggle.com/datasets/milanvaddoriya/old-car-price-prediction) yang terdiri dari total 5511 entri data. Dataset ini dipublikasikan oleh [Millan Vaddoriya](https://www.kaggle.com/milanvaddoriya) yang melakukan scrapping data dari https://www.cardekho.com/used-car-details.


### 2.1 Variabel-variabel pada Old Car Price Prediction dataset :
Data yang digunakan dalam proyek ini adalah dataset yang bersumber dari kaggle bernama `Old Car Price Prediction`. Dataset ini terdiri dari 9 kolom diantaranya
1. `car_name` yang berisi data nama kendaraan
2. `car_prices_in_rupee` yang berisi data nominal harga dari kendaraan dalam rupee dimana terdapat harga dengan nominal `Lakh` yang sama dengan konstanta 100.000 (seratus ribu) dan `Crore` sama dengan konstatnta 10.000.000 (sepuluh juta).
3. `kms_driven` yang berisi data jumlah kilometer yang telah ditempuh
4. `fuel_type` yang berisi data tipe bahan bakar kendaraan
5. `transmission` yang berisi data tipe transmisi kendaraan
6. `ownership` yang berisi data jumlah pemilik sebelumnya dimulai dari 1st Owner - 5th Owner
7. `manufacture` yang berisi data tahun pembuatan kendaraan
8. `engine` yang berisi data kapasitas mesin kendaraan dalam cc
9. `Seats` yang berisi data kapasitas tempat duduk dalam kendaraan dimulai dari 2 Seats - 8 Seats

Untuk memahami data dilakukan dengan menggunakan teknik Exploratory Data Analysis (EDA).Sebelum melakukan proses EDA, terlebih dahulu dilakukan proses berupa Data Loading dan Data Cleaning. 

### 2.2 Data Loading
Data loading dilakukan untuk mengunduh data dari kaggle dengan cara mengunggah kredensial akun kaggle berupa file kaggle.json ke dalam direktori proyek. Kemudian, dilakukan serangkaian proses untuk mengkonfigurasi kresensial kaggle yang telah diupload. Langkah selanjutnya adalah mengunduh dataset melalui perintah `!kaggle datasets download milanvaddoriya/old-car-price-prediction` dan dilanjutkan dengan melakukan unzip dataset yang telah didownload.

### 2.3 Data Cleaning
Pada proses Data Cleaning dilakukan proses pembersihan data dengan menghilangkan beberapa atribut dalam tiap-tiap entri data. Detail dari proses pembersihan data adalah sebagai berikut :
1. Melakukan dropping kolom `unnamed`.
2. Melakukan splitting pada entri data di kolom `car_prices_in_rupee` dan memindahkan hasilnya pada kolom baru bernama `price` yang berisi angka dari setiap entri data dan kolom `multiply` yang berisi atribut dari setiap entri data (Lakh atau Crore).
3. Mengubah entri Lakh pada kolom `multiply` menjadi nilai 100000 dan entri Crore menjadi nilai 10000000.
4. Mengalikan setiap entri data pada kolom `price` dengan `multiply` serta menyimpan hasilnya pada kolom `price`. Akhirnya akan didapatkan data pada kolom `price` yang berisi data harga bertipe numerik.
5. Menghilangkan atribut "kms" pada entri data `kms_driven` dan menghilangkan tanda koma pada masing-masing entry data.
6. Mengubah entri data pada kolom `price` yang tidak berisi number (NAN) menjadi bialngan nol.
7. Melakukan splitting pada setiap entri data di kolom `car_name` dimana hanya akan diambil nama pabrikan mobil saja yang akan disimpan pada kolom baru bernama `manufacturer`. Selain itu, akan diambil juga nama model kendaraan yang akan disimpan pada kolom baru bernama `Model`.
8. Menghilangkan atribut `st Owner`, `nd Owner`, `rd Owner`, `th Owner`pada kolom `ownership` sehingga kita hanya akan mendapatkan data berupa angka saja.
9. Menghapus atribut `cc` pada kolom `engine`.
10. Menghapus atribut `Seats` pada kolom `Seats`.
11. Mengganti nama kolom `manufacture` menjadi `year` agar memudahkan dalam membaca data dan menghindari kesalahan.
12. Memeriksa tipe data masing-masing fitur dan mengonversinya ke tipe data yang sesuai untuk proses EDA.

### 2.4 Exploratory Data Analysis
Dalam tahapan ini akan dilakukan analisis data yang sudah dibersihkan meliputi:
#### 2.4.1. Deskripsi variabel
Tahapan ini dilakukan untuk mengetahui informasi variabel dari dataset diantaranya jumlah kolom, jumlah data, nama kolom, jumlah data per kolom, dan tipe data.

#### 2.4.2. Deskripsi Statistik
Tahapan ini dilakukan untuk mengetahui deskripsi statistik dari dataset. Dimana pada tahapan ini kita akan mendapatkan deskripsi statistik meliputi :
1. Count : Jumlah sampel data
2. Mean : Nilai rerata
3. STD : Standar Deviasi
4. Min : Nilai minimum
5. 25% : Kuartil bawah atau Q1
6. 50% : Kuartil tengah atau Q2 atau Median
7. 75% : Kuartil atas / Q3
8. Max : Nilai Maksimum

Berdasarkan deskripsi statistik dari dataset, kita dapat mengetahui apakah terdapat missing value atau tidak dengan melihat apakah terdapat nilai minimum yang bernilai nol. 

#### 2.4.3. Menangani Missing Value
Dalam tahapan ini, akan dilakukan dropping pada setiap data yang memiliki missing value.

#### 2.4.4. Menangani Outliers
Dalam tahapan ini akan dilakukan pengecekkan outliers menggunakan teknik visualisasi data dengan boxplot. Apabila terdapat outliers, maka akan ditangani dengan menggunakan teknik Inter Quartile Range (IQR).Seltman dalam “Experimental Design and Analysis” [24] menyatakan bahwa outliers yang diidentifikasi oleh boxplot (disebut juga “boxplot outliers”) didefinisikan sebagai data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1.

Berdasarkan teori tersebut, kita dapat membuat batas atas dan batas bawah dari data:

- Batas bawah = Q1 - 1.5 * IQR
- Batas atas = Q3 + 1.5 * IQR

#### 2.4.5. Melakukan Univariate Analysis
Univariate Analysis dalam Exploratory Data Analysis (EDA) merujuk pada jenis analisis yang dilakukan terhadap satu variabel atau fitur tunggal dalam dataset, tanpa mempertimbangkan hubungannya dengan variabel lain.Dalam tahapan ini akan dilakukan visualisasi menggunakan histogram dari masing-masing fitur yaitu `categorical_features` dan `numerical_features`.

#### 2.4.6. Melakukan Multivariate Analysis
Multivariate Analysis dalam Exploratory Data Analysis (EDA) adalah jenis analisis yang dilakukan terhadap dua atau lebih variabel dalam dataset, dengan tujuan untuk memahami hubungan, interaksi, dan pola kompleks antara variabel-variabel tersebut. 

Pada `categorical_features`, analisis dilakukan dengan membuat grafik rata-rata `price` relatif terhadap masing-masing fitur kategori.

Sedangkan, pada `numerical_features`, analisis dilakukan dengan menggunakan fungsi `pairplot()` untuk melakukan observasi secara visual dan dilanjutkan menggunakan fungsi `corr()` untuk mengetahui skor korelasinya.


## 3. Data Preparation
Pada tahapan ini kita akan melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Terdapat 4 tahapan umum dalam Data Preparation diantaranya :
1. Encoding Fitur Kategori
2. Reduksi dimensi dengan PCA
3. Pembagian dataset dengan fungsi train_test_split dari library `sklearn`
4. Standardisasi

### 3.1 Encoding Fitur Kategori
Untuk melakukan encoding fitur kategori, salah satu teknik yang umum digunakan adalah teknik **one-hot-encoding**. Library Sklearn menyediakan fungsi ini untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori. Dalam dataset ini kita memiliki 4 fitur kategori yaitu  `model`, `fuel_type`, `transmission`, dan `manufacturer`. Kita akan melakukan proses encoding ini dengan fitur `get_dummies`.

### 3.2 Reduksi Dimensi dengan PCA
Teknik reduksi merupakan prosedut untuk mengurangi sejumlah fitur dengan tetap mempertahankan informasi pada data. Teknik reduksi yang paling populer adalah principle component analysis (PCA). Teknik ini digunakan untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari "n-dimensional space" ke dalam sistem berkoordinat baru dengan dimensi m, dimana m lebih kecil dari n.

PCA ini bekerja menggunakan metode aljabar linier. Dimana ia mengasumsikan bahwa sekumpulan data pada arah dengan varians terbesar merupakan yang paling penting atau utama. PCA umumnya digunakan ketika variabel dalam data memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. Oleh karena itu, PCA digunakan untuk mereduksi variabel asli menjadi sejumlah kecil variabel baru yang tidak berkorelasi linier. Komponen utama dapat menangkap sebagian besar varians dalam variabel asli. Sehingga PCA yang diterapkan pada data, hanya akan menggunakan komponen utama dan mengabaikan sisanya.

Seperti yang kita tahu ketika beberapa fitur memiliki korelasi yang tinggi terhadap price, berarti fitur tersebut memuat informasi yang sama. Contohnya Tahun Pembuatan dan Kilometer kendaraan yang memuat informasi terkait masa pakai kendaraan.

Setelah melakukan PCA, akan didapatkan proporsi informasi dari keenam komponen. Dimana akan dalam dataset ini akan berfokus pada principle component yang pertama.

### 3.3 Train-Test-Split

Proporsi pembagian data train dan test bisanya adalah 80:20. Tujuan dari adanya data test adalah untuk mengukur kinerja model pada data baru. Pada proyek ini kita akan melakukan proporsi pembagian sebesar 90:10 dengan menggunakan fungsi train_test_split dari sklearn.

### 3.4 Standardisasi
Standardisasi adalah teknik transformasi yang umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one hot-encidng seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library sklearn.

StarndardScaler melakukan proses standardisasi dengan mengurangi mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler ini menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Kita hanya akan menerapkan fitur standardisasi pada data latih saja untuk menghindari kebocoran informasi pada data uji. Barulah pada tahap evaluasi kita akan melakukan standardisasi pada data uji.



## 4. Modeling
Dalam tahapan ini kita akan mengembangkan machine learning dengan tiga algoritma dan akan dilakukan evaluasi pada ketiganya untuk mengetahui algoritma mana yang dapat memberikan hasil prediksi terbaik. ALgoritma yang digunakan diantaranya:
1. K-Nearest Neighbor
2. Random Forest
3. Boosting Algorithm

### 4.1 Algoritma K-Nearest Neighbor (KNN)

Algoritma ini akan bekerja pada data baru dimana setiap data baru akan diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).

Memilih nilai k yang lebih besar dapat membantu menghindari overfit, meskipun terkadang menyebabkan kehilangan kemampuan prediksi. Ketika memilih nilai k=1 maka hasilnya akan sangat kaku sedangkan nilai k yang besar hasil prediksi akan lebih halus.

- k terlalu rendah = overfit, hasil prediksi varians nya tinggi
- k terlalu tinggi = underfit, prediksi memiliki bias yang tinggi

Untuk menentukan titik mana dalam data yang paling mirip dengan input baru, KNN menggunakan perhitungan ukuran jarak diman ametrik yang digunakan secara default pada library sklearn adalah Minkowski Distance [27] metrik lain yang sering dipakai adalah Euclidean dan Manhattan distance. Dalam proyek ini akan dipakai metric default di library sklearn untuk KNN yaitu Euclidean Distance.

### 4.2 Algoritma Random Forest

Algortima ini merupakan salah satu model machine learning berkategori ensemble (group) learning. Maksudnya yaitu model ini terdiri dari beberapa model yang bekerja secara bersama-sama sehingga tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada ensemble, tiap model harus membuat prediksi secara independen, lalu hasilnya akan digabungkan untuk membuat prediksi akhir.

Untuk membuat model ensemble ada dua teknik pendekatan:
1. Bagging atau Bootstrap Aggregating
2. Boosting

`Bagging atau Bootstrap aggregating` yaitu teknik yang melatih model dengan sampel random. Dalam teknik ini model dilatih dengan teknik proses sampling dengan penggantian. Sampel dengan nilai yang berbeda bersifat independen, artinya nilai suatu sampel tidak mempengaruhi sampel lainnya [28]. Akibatnya model yang dilatih akan berbeda antara satu dan lainnya.

Salah satu algoritma yang cocok dengan teknik bagging adalah decision tree. Misalnya, ada 200 model decision tree pada bag random forest kita, ini berarti bahwa keputusan (decision) yang dibuat oleh setiap pohon (model) akan sangat bervariasi. Pada kasus klasifikasi, prediksi akhir diambil dari prediksi terbanyak pada seluruh pohon. Sedangkan, pada kasus regresi, prediksi akhir adalah rata-rata prediksi seluruh pohon dalam model ensemble.Maka algoritma ini disebut random forest karena tersusun dari banyak algoritma decision tree yang pembagian data dan fiturnya dipilih secara acak.

Untuk menggunakan algoritma diharuskan mengimpor `RandomForestRegressor` dari `sklearn.ensemble` yang memiliki nilai parameter yang perlu diatur diantaranya :
- `n_estimator`: jumlah trees 
- `max_depth`: panjang atau kedalaman pohon adalah ukuran seberapa banyak pohon dapat splitting untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan
- `random_state`. digunakan untuk mengontrol random number generator yang digunakan
- `n_jobs`: jumlah job yang digunakan secara paralel yaitu komponen untuk mengontrol thread atau proses yang berjalan secara paralel. Nilai -1 artinya semua proses berjalan secara paralel.

### 4.3 Boosting Algorithm

Boosting merupakan model ensemble yang melatih model secara berurutan (tidak paralel). Ia akan membangun model dari data latih. Kemudian, ia akan membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model terus ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

**Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi. Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest.***

Dilihat dari caranya memperbaiki kesalahan pada model sebelumnya, algoritma boosting terdiri dari dua metode:

- Adaptive boosting
- Gradient boosting

**Adaptive boosting** adalah metode boosting yang memperbaiki kesalahan pada model sebelumnya dengan memberikan bobot yang lebih tinggi pada data yang salah diklasifikasikan. Bobot ini akan terus diperbarui di setiap iterasi, sehingga model yang dihasilkan akan semakin akurat.

Pada iterasi pertama, model lemah akan dilatih pada data dengan bobot yang sama. Model ini akan memberikan prediksi untuk setiap data. Data yang salah diklasifikasikan akan diberikan bobot yang lebih tinggi pada iterasi selanjutnya.

Pada iterasi selanjutnya, model lemah baru akan dilatih pada data dengan bobot yang diperbarui. Model ini akan dirancang untuk memperbaiki kesalahan yang dilakukan oleh model lemah sebelumnya.

Proses ini akan terus berulang-ulang hingga kesalahan pada model sebelumnya dapat diminimalkan.

**Gradient boosting** adalah metode boosting yang memperbaiki kesalahan pada model sebelumnya dengan menambahkan model baru yang dirancang untuk mengurangi gradien loss function. Loss function adalah fungsi yang mengukur kesalahan model.

Pada iterasi pertama, model lemah akan dilatih pada data. Model ini akan memberikan prediksi untuk setiap data.

Pada iterasi selanjutnya, model baru akan dilatih pada data yang salah diklasifikasikan oleh model lemah sebelumnya. Model baru ini akan dirancang untuk mengurangi gradien loss function.

Proses ini akan terus berulang-ulang hingga gradien loss function dapat diminimalkan.


AdaBoost dan gradient boosting memiliki beberapa perbedaan, antara lain:

- `Cara memperbaiki kesalahan`: AdaBoost memperbaiki kesalahan dengan memberikan bobot yang lebih tinggi pada data yang salah diklasifikasikan. Gradient boosting memperbaiki kesalahan dengan menambahkan model baru yang dirancang untuk mengurangi gradien loss function.
-`Model yang digunakan`: AdaBoost dapat menggunakan berbagai model lemah, seperti pohon keputusan, regresi linear, dan jaringan saraf tiruan. Gradient boosting biasanya menggunakan pohon keputusan sebagai model lemah.
- `Performance`: AdaBoost umumnya lebih cepat daripada gradient boosting.


Pada proyek ini kita akan menggunakan metode adaptive boosting yaitu AdaBoost. Awalnya, semua kasus dalam data latih memiliki weight atau bobot yang sama. Pada setiap tahapan, model akan memeriksa apakah observasi yang dilakukan sudah benar? Bobot yang lebih tinggi kemudian diberikan pada model yang salah sehingga mereka akan dimasukkan ke dalam tahapan selanjutnya. Proses iteratif ini berlanjut sampai model mencapai akurasi yang diinginkan.

## 5. Evaluation
Metrik yang digunakan dalam prediksi ini adalah MSE yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Berikut merupakan formula dari MSE

![link text](https://miro.medium.com/v2/resize:fit:1198/1*BtVajQNj29LkVySEWR_4ww.png
)

Sebelum melakukan perhitungan MSE, perlu dilakukan scaling pada fitur numerik pada data uji. Setelah model selesai kita latih dengan 3 algoritma, kita perlu melakukan proses scaling terhadap data uji. Hal tersebut dilakukan agar skala antara data latih dan data uji sama dan bisa kita lakukan evaluasi.

Evaluasi model dilakukan dengan menghitung Mean Squarred Error dari masing-masing algoritma pada data train dan test. Hasil perhitungan tersebut akan ditampilkan dalam bentuk bar chart untuk memudahkan proses evaluasi. Model yang dipilih merupakan model yang memberikan nilai error paling kecil pada data train dan test.

Proses evaluasi dilanjutkan dengan membuat prediksi menggunakan beberapa entri data di kolom `price` dari dataset dan menampilkan hasil R2 Score pada masing-masing model. R2 menunjukkan seberapa dekat hasil prediksi dengan data sebenarnya.  Model yang memiliki R2 Score atau koefisien determinasi tertinggi merupakan model yang terbaik.

## 6. Daftar Pustaka
[27] Scikit-learn Documentation. Tersedia: tautan. Diakses pada: Juni 2021.

[28] Parker, Mary. "Sampling with Replacement and Sampling without Replacement". Tersedia: tautan. Diakses pada Juni 2021. 



