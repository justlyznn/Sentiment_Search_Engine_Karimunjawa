# 🏖️ Search Engine Pantai Karimunjawa

Aplikasi web interaktif untuk mengeksplorasi dan menganalisis review pantai di Karimunjawa dengan kemampuan pencarian cerdas berdasarkan rating, sentimen, dan kata kunci.

## 📋 Deskripsi Aplikasi

Aplikasi ini memungkinkan pengguna untuk:
- 🔍 **Mencari review** pantai di Karimunjawa dengan filter yang fleksibel
- ⭐ **Menganalisis rating** dari berbagai pantai
- 😊 **Memfilter berdasarkan sentimen** (positif, netral, negatif)
- 🏆 **Melihat ranking pantai terbaik**
- 📊 **Mendapatkan rekomendasi** berdasarkan kategori sentimen

## 🚀 Fitur Utama

### 1. **Pencarian Cerdas**
   - Filter berdasarkan rating (1-5 bintang)
   - Filter berdasarkan kategori sentimen
   - Pencarian kata kunci dalam review
   - Pencarian real-time dengan preprocessing teks

### 2. **Visualisasi Data**
   - **TOP 3 Pantai Terbaik** dengan ranking emas, perak, perunggu
   - **5 Pantai Rekomendasi** per kategori sentimen
   - Statistik ringkasan (rating rata-rata, review positif/negatif)
   - Tampilan tabel review dengan detail lengkap

### 3. **User Interface yang Intuitif**
   - Sidebar untuk semua filter
   - Tab untuk navigasi kategori sentimen
   - Tombol interaktif dengan ikon minimalis
   - Tampilan responsif dan modern

## 🛠️ Teknologi yang Digunakan

- **Python 3.x**
- **Streamlit** - Framework untuk aplikasi web interaktif
- **Pandas** - Manipulasi dan analisis data
- **NumPy** - Komputasi numerik
- **Regular Expressions** - Preprocessing teks

## 📊 Struktur Data

Dataset utama (`data/labelling/label_data.csv`) harus memiliki struktur:

```csv
title,stars,text,sentimen
Pantai Tanjung Gelam,5,"Pantai yang sangat indah...",positive
Pantai Anora,3,"Cukup bagus tapi ramai",neutral
...
```

**Keterangan kolom:**
- `title`: Nama pantai (string)
- `stars`: Rating 1-5 (integer)
- `text`: Isi review (string)
- `sentimen`: Kategori sentimen - positive/neutral/negative (string)

## 🚀 Cara Menjalankan Aplikasi

### 1. **Persiapan Environment**
```bash
# Clone repository
git clone <repository-url>

# Masuk ke direktori project
cd karimunjawa-search-engine

# Install dependencies
pip install -r requirements.txt
```

### 2. **Setup Data**
Pastikan dataset tersedia di `data/labelling/label_data.csv`. Jika belum:

```bash
# Salin data ke folder yang benar
mkdir -p data/labelling
cp label_data.csv data/labelling/label_data.csv
```

### 3. **Menjalankan Aplikasi**
```bash
streamlit run app/app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## 🔧 Pipeline Data

### **1. Data Collection (Raw)**
- Data review dikumpulkan dari berbagai sumber
- Disimpan di `data/raw/reviews_raw.csv`

### **2. Data Cleaning**
```python
# data/cleaning/clean_data.py
- Hapus duplikat
- Handle missing values
- Standardize format
```

### **3. Sentiment Labelling**
```python
# data/labelling/label_data.py
- Klasifikasi manual/otomatis
- Label: positive/neutral/negative
- Hasil: label_data.csv
```

### **4. Text Preprocessing**
```python
# data/preprocessing/preprocess.py
- Lowercasing
- Remove special characters
- Tokenization (opsional)
```

## 📊 Algoritma Ranking

### **TOP 3 Pantai Terbaik**
Skor ranking dihitung dengan formula:
```
Score = Rating_Rata2 × (Persentase_Positif / 100) × log(1 + Jumlah_Review)
```

### **5 Pantai Rekomendasi per Sentimen**
```
Score_Sentimen = Rating_Rata2 × log(1 + Jumlah_Review)
```

## 🎨 UI/UX Features

### **Sidebar Filter**
- Slider rating (1-5 bintang)
- Dropdown pilihan sentimen
- Input field pencarian kata kunci

### **Main Dashboard**
- **Bagian 1:** Visual ranking TOP 3 dengan badge emas, perak, perunggu
- **Bagian 2:** 5 pantai rekomendasi per tab sentimen
- **Bagian 3:** Tabel review detail dengan kemampuan sorting

### **Interaksi**
- Klik tombol `>` untuk melihat review detail
- Klik tombol `✖️` untuk menutup tabel review
- Semua filter bekerja secara real-time

## 📈 Use Cases

### **Untuk Wisatawan**
- Mencari pantai dengan rating terbaik
- Membaca review dari pengunjung lain
- Memfilter berdasarkan preferensi (sunset, bersih, ramah)

### **Untuk Pengelola Wisata**
- Menganalisis sentimen pengunjung
- Mengidentifikasi area perbaikan
- Melihat perbandingan antar pantai

### **Untuk Peneliti**
- Analisis data review pariwisata
- Studi sentimen natural language processing
- Visualisasi data geografis

## 🧪 Testing

### **Unit Testing**
```python
# Contoh test untuk fungsi preprocessing
def test_preprocess_text():
    assert preprocess_text("Pantai Indah!") == "pantai indah "
    assert preprocess_text(None) == ""
```

### **Data Validation**
- Validasi format CSV
- Cek missing values
- Validasi range rating (1-5)

## 🐛 Troubleshooting

### **Masalah Umum**

1. **File CSV tidak ditemukan**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'data/labelling/label_data.csv'
   ```
   **Solusi:** Pastikan struktur folder sesuai dan file ada di lokasi yang benar

2. **Module tidak ditemukan**
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solusi:** Install dependencies dengan `pip install -r requirements.txt`

3. **Aplikasi lambat**
   - Gunakan `@st.cache_data` untuk caching
   - Optimasi query pandas
   - Batasi jumlah data yang ditampilkan

## 📱 Responsive Design

Aplikasi mendukung berbagai ukuran layar:
- **Desktop:** Tampilan 5 kolom untuk rekomendasi
- **Tablet:** Tampilan 3 kolom
- **Mobile:** Tampilan 1 kolom

## 🔒 Security Considerations

- **Input Validation:** Semua input user divalidasi
- **Data Sanitization:** Preprocessing teks untuk mencegah XSS
- **Caching:** Implementasi caching untuk performa

## 📄 License

Proyek ini menggunakan lisensi MIT.

## 👥 Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## ✨ Tips Penggunaan

1. **Untuk pencarian spesifik:** Gunakan kombinasi filter rating + sentimen + kata kunci
2. **Untuk analisis cepat:** Gunakan tab sentimen untuk fokus pada kategori tertentu
3. **Untuk perbandingan:** Perhatikan skor ranking di TOP 3 untuk pantai terbaik

## 🔮 Future Enhancements

- [ ] Integrasi peta lokasi pantai
- [ ] Analisis trend waktu
- [ ] Export data ke Excel/PDF
- [ ] Dashboard admin untuk upload data
- [ ] Integrasi dengan API weather


---

**Dibuat dengan ❤️ untuk para pencinta pantai Karimunjawa**  
*Semoga aplikasi ini membantu menemukan pantai impian Anda!*