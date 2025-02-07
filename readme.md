# Laporan Proyek Machine Learning - Anime Recommendation System

## Project Overview
_Recommendation system_ adalah sebuah sistem yang mengacu pada memprediksi sejumlah item atau data untuk pengguna di masa lalu, kemudian dijadikan rekomendasi item paling teratas.

## Business Understanding

### Problem Statements

Setiap konten streaming memiliki penontonnya sendiri dan setiap konten memiliki ratingnya sendiri. Penonton memberikan rating yang bagus untuk konten tersebut jika mereka menyukainya. Namun, di mana hal ini berlaku? Penonton dapat menghabiskan waktu berjam-jam menelusuri ratusan, terkadang ribuan anime, dan tidak pernah menemukan konten yang mereka sukai.

Sehingga, bagaimana cara memberikan rekomendasi anime yang disukai oleh pengguna?

### Goals
Untuk menyelesaikan permasalahan yang telah disampaikan pada bagian _Problem Statement_, maka dibuat sistem rekomendasi yang dapat memberikan rekomendasi film berdasarkan _ratings_.

### Solution statements
Solusi pembuatan model yang dilakukan adalah dengan menerapkan 1 algoritma machine learning, terbatas pada **Collaborative Filtering**. Algoritma collaborative filtering akan merekomendasikan pengguna berdasarkan rating yang paling tinggi.

- **Collaborative Filtering**
Algoritma Collaborative Filtering adalah algoritma yang menggunakan kesamaan antara pengguna dan item secara bersamaan untuk memberikan rekomendasi. Algoritma tersebut juga bergantung pada preferensi pengguna serupa untuk menawarkan rekomendasi kepada pengguna tertentu.

## Data Understanding
Dataset yang digunakan pada proyek _machine learning_ merupakan **17.562 data anime** dan **data preferensi dari 325.772 user** yang didapat dari situs [Kaggle](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020). 

**File dan Variabel dari Anime Recommendation Database 2020 adalah sebagai berikut:**

1.  animelist.csv
    - user_id: ID pengguna yang dibuat secara acak dan tidak dapat diidentifikasi.
    - anime_id: ID MyAnimeList dari anime tersebut.
    - score: skor antara 1 hingga 10 yang diberikan oleh pengguna. 0 jika pengguna tidak memberikan skor.
    - watching_status: ID status dari anime ini dalam daftar anime pengguna ini.
    - watched_episodes: jumlah episode yang ditonton oleh pengguna.

2.  watching_status.csv = Menjelaskan setiap kemungkinan status kolom: "watching_status" di animelist.csv.
3.  rating_complete.csv
    - user_id: ID pengguna yang dibuat secara acak dan tidak dapat diidentifikasi.
    - anime_id: ID MyAnimeList dari anime tersebut.
    - rating: Peringkat yang telah diberikan oleh pengguna ini.

4. anime.csv = Berisi informasi umum setiap anime (17.562 anime berbeda) seperti genre, statistik, studio, dll. File ini memiliki kolom-kolom berikut:
    - MAL_ID: ID MyAnimelist dari anime tersebut.
    - Name: nama lengkap animenya.
    - Score: skor rata-rata anime yang diberikan dari semua pengguna di basis data MyAnimelist.
    - Genres: daftar genre yang dipisahkan koma untuk anime ini.
    - English name: nama lengkap anime dalam bahasa inggris.
    - Japanese name: nama lengkap anime dalam bahasa jepang.
    - Type: TV, film, OVA, dll.
    - Episodes': jumlah bab.
    - Aired: tanggal siaran.
    - Premiered: musim perdana.
    - Producers: daftar produsen yang dipisahkan koma.
    - Licensors: daftar pemberi lisensi yang dipisahkan koma.
    - Studios: daftar studio yang dipisahkan koma.
    - Source: Manga, Novel ringan, Buku, dll.. 
    - Duration: durasi anime per episode.
    - Rating: tingkat usia 
    - Ranked: posisi berdasarkan skor.
    - Popularity: posisi berdasarkan jumlah pengguna yang telah menambahkan anime ke daftar mereka.
    - Members: jumlah anggota komunitas yang ada di "grup" anime ini.
    - Favorites: jumlah pengguna yang menjadikan anime tersebut sebagai "favorit".
    - Watching: jumlah pengguna yang menonton anime.
    - Completed: jumlah pengguna yang telah menyelesaikan anime. 
    - On-Hold: jumlah pengguna yang menahan anime. 
    - Dropped: jumlah pengguna yang telah menghentikan anime.
    - Plan to Watch': jumlah pengguna yang berencana menonton anime.
    - Score-10': jumlah pengguna yang mendapat skor 10.
    - Score-9': jumlah pengguna yang mendapat skor 9. 
    - Score-8': jumlah pengguna yang mendapat skor 8. 
    - Score-7': jumlah pengguna yang mendapat skor 7. 
    - Score-6': jumlah pengguna yang mendapat skor 6. 
    - Score-5': jumlah pengguna yang mendapat skor 5. 
    - Score-4': jumlah pengguna yang mendapat skor 4. 
    - Score-3': jumlah pengguna yang mendapat skor 3. 
    - Score-2': jumlah pengguna yang mendapat skor 2. 
    - Score-1': jumlah pengguna yang mendapat skor 1. 


### Explanatory Data Analysis
Dari list file yang terdapat dari dataset, sistem rekomendasi ini hanya menggunakan data dari `anime.csv` dan `animelist.csv`.

Dari analisis yang telah dilakukan pada kode yang telah dibuat, data dari `anime.csv` dan `animelist.csv` dapat dikatakan bersih karena tidak memiliki _missing values_.

## Data Preparation

Tahapan _data preparation_ yang dilakukan meliputi:

1. Menghapus baris duplikat: Dataset `rating` diperiksa untuk mendeteksi baris duplikat menggunakan `duplicated()`. Jika ditemukan baris duplikat, baris tersebut akan dihapus dari dataset untuk memastikan bahwa data yang digunakan bersih dan bebas duplikasi.

2. Menyaring Pengguna Berdasarkan Jumlah Rating: Hanya pengguna yang telah memberikan setidaknya 350 rating pada anime yang dipertahankan dalam dataset. Hal ini dilakukan untuk memastikan bahwa data memiliki kontribusi yang cukup signifikan terhadap proses pelatihan model.

3. Normalisasi Rating: _Rating_ pada dataset dinormalisasi menggunakan metode `Min-Max Scaling` untuk memastikan nilainya berada dalam rentang 0 hingga 1. Langkah ini membantu model mempelajari pola dengan lebih baik karena data berada pada skala yang seragam.

4. Encoding: Kolom kategori `user_id` dan `anime_id` dikonversi menjadi format numerik untuk mempermudah proses pelatihan model. Proses ini dilakukan menggunakan dictionary mapping.

5. Mengacak Dataset (Shuffling): Dataset diacak menggunakan `sample(frac=1)` untuk menghindari pola tertentu pada data yang dapat memengaruhi performa model. Fitur (X) dan target (y) dipisahkan untuk keperluan pelatihan model.


6. Train-Test Split: Dataset dibagi menjadi dua bagian dengan rasio 80:20, di mana 80% digunakan untuk training set dan 20% untuk testing set.


## Modeling
Tahapan ini membahas mengenai model deep learning berbasis Collaborative Filtering yang digunakan untuk membangun sistem rekomendasi anime. Sistem ini memanfaatkan embedding untuk merepresentasikan hubungan antara pengguna dan anime berdasarkan rating. Model dibuat menggunakan framework TensorFlow/Keras.

1. Arsitektur Model: Model ini dirancang menggunakan arsitektur neural network dengan lapisan embedding untuk mewakili pengguna dan anime. Berikut detail arsitektur yang digunakan:
    - Input Layer
        - User Input: Input ID pengguna dengan bentuk [1], digunakan untuk mendapatkan representasi embedding pengguna.
        - Anime Input: Input ID anime dengan bentuk [1], digunakan untuk mendapatkan representasi embedding anime.
    - Embedding Layer
        - User Embedding: `Embedding(name='user_embedding', input_dim=n_users, output_dim=embedding_size)`.
        - Anime Embedding: E`mbedding(name='anime_embedding', input_dim=n_animes, output_dim=embedding_size)`.
    - Dot Product
    Lapisan `Dot` digunakan untuk menghitung tingkat kesamaan (similarity) antara embedding pengguna dan anime melalui operasi dot product. Normalisasi dilakukan untuk menstandardisasi hasil.
    - Fully Connected Layer
    Hasil dari Dot kemudian diproses melalui lapisan dense, yaitu:
        - Dense Layer: Memberikan skor dengan inisialisasi kernel He (kernel_initializer='he_normal').
        - Batch Normalization: Menormalkan output lapisan dense untuk mempercepat konvergensi.
        - Activation: Menggunakan fungsi aktivasi sigmoid untuk memastikan output berada pada skala 0-1.
    - Output Layer
    Lapisan terakhir menghasilkan skor prediksi (rating) antara pengguna dan anime dengan skala probabilitas (0 hingga 1).
    - Kompilasi Model
    Model dikompilasi dengan parameter sebagai berikut:
        - Loss Function: `binary_crossentropy` digunakan untuk mengukur kesalahan prediksi dalam skala biner.
        - Optimizer: `Adam` digunakan untuk mempercepat proses training.
        - Metrics: Evaluasi menggunakan Mean Absolute Error (MAE) dan Mean Squared Error (MSE)

2. Callback
Beberapa callback digunakan untuk meningkatkan efisiensi dan performa selama proses pelatihan:
    - EarlyStopping: Menghentikan training jika validasi loss tidak membaik setelah 5 epoch.
    - ModelCheckpoint: Menyimpan model dengan validasi loss terbaik.
    - ReduceLROnPlateau: Mengurangi learning rate jika validasi loss stagnan.
    - TensorBoard: Merekam log pelatihan untuk visualisasi.
3. Training
    - Input Data: Model dilatih menggunakan data kombinasi `[X_train[:, 0], X_train[:, 1]]` (ID pengguna dan anime) sebagai input dan `y_train` (rating) sebagai label.
    - Validation Data: Validasi dilakukan menggunakan `[X_test[:, 0], X_test[:, 1]]` sebagai input dan `y_test` sebagai label.
    - Epoch: Maksimal 50 epoch digunakan, tetapi training dapat berhenti lebih awal jika syarat callback terpenuhi.
    - Batch Size: Menggunakan batch size sebesar 256 untuk meningkatkan efisiensi komputasi.


## Evaluation
Kode ini digunakan untuk memberikan rekomendasi anime kepada pengguna berdasarkan data yang telah diolah menggunakan model pembelajaran. Metrik evaluasi yang digunakan untuk menilai kinerja model adalah **Root Mean Squared Error (RMSE)**. RMSE mengukur sejauh mana perbedaan antara nilai prediksi model dengan nilai aktual yang diamati. Nilai ini diperoleh dari akar kuadrat dari **Mean Squared Error (MSE)**. Semakin kecil nilai RMSE, semakin akurat model dalam melakukan estimasi, yang menunjukkan bahwa kesalahan prediksi model lebih rendah.

Hasil dari evaluasi matriks adalah sebagai berikut:

![output](https://github.com/user-attachments/assets/dac15cab-0548-40d6-bc68-4968e913ddc5)



Berdasarkan hasil visualisasi proses pelatihan model, terlihat bahwa model mulai mengalami overfitting. Hal ini dapat dilihat dari perbedaan tren antara loss data train dan test.

Pada awalnya, baik loss data train maupun test menurun seiring bertambahnya epoch. Namun, setelah beberapa epoch, loss data test mulai meningkat, sementara loss data train terus menurun secara konsisten. Ini menunjukkan bahwa model terlalu menyesuaikan diri dengan data train sehingga kehilangan kemampuan generalisasi terhadap data baru (test set).

Dari grafik ini, nilai loss data train terakhir sekitar 0.42, sedangkan loss data test berada di atas 0.56. Perbedaan ini mengindikasikan bahwa model tidak bekerja optimal pada data test, sehingga diperlukan langkah-langkah seperti penambahan regularization, early stopping, atau pengurangan kompleksitas model untuk mengatasi overfitting.
