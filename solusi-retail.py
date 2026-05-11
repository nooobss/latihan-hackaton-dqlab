import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# Mengabaikan warning agar output terminal bersih
warnings.filterwarnings('ignore')

def main():
    print("Memulai proses analisis data...")
    
    # ==========================================
    # LANGKAH 1: PERSIAPAN DATA
    # ==========================================
    # Silakan sesuaikan nama file dengan yang Anda miliki (.csv atau .xlsx)
    try:
        df = pd.read_csv('data_penjualan.csv')
    except FileNotFoundError:
        df = pd.read_excel('data_penjualan.xlsx')
        
    df['tgl_transaksi'] = pd.to_datetime(df['tgl_transaksi'], format='%d-%m-%Y')
    
    # ==========================================
    # LANGKAH 2: IDENTIFIKASI RISING STAR
    # ==========================================
    # 1. Agregasi Penjualan Harian per Produk
    daily_sales = df.groupby(['tgl_transaksi', 'kode_produk', 'nama_produk'])['total_nilai'].sum().reset_index()
    daily_sales = daily_sales.sort_values(by=['kode_produk', 'tgl_transaksi'])
    
    # 2. Perhitungan Moving Average 3 Hari
    daily_sales['MA_3'] = daily_sales.groupby('kode_produk')['total_nilai'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # 3. Menandai Tren Naik (Sesi Tren Naik)
    daily_sales['MA_prev'] = daily_sales.groupby('kode_produk')['MA_3'].shift(1)
    daily_sales['is_rising'] = daily_sales['MA_3'] > daily_sales['MA_prev']
    
    # 4. Mencari 12 Hari Berturut-turut
    daily_sales['grup_tren'] = (~daily_sales['is_rising']).groupby(daily_sales['kode_produk']).cumsum()
    tren_naik = daily_sales[daily_sales['is_rising'] == True]
    
    durasi_tren = tren_naik.groupby(['kode_produk', 'nama_produk', 'grup_tren']).size().reset_index(name='consecutive_days')
    rising_star_candidates = durasi_tren[durasi_tren['consecutive_days'] >= 12]
    
    # 5. Menghitung Growth % dan Menyusun DataFrame Rising Star
    hasil_rising_star = []
    for index, row in rising_star_candidates.iterrows():
        kode = row['kode_produk']
        nama = row['nama_produk']
        grup = row['grup_tren']
        
        data_sesi = tren_naik[(tren_naik['kode_produk'] == kode) & (tren_naik['grup_tren'] == grup)]
        
        ma_akhir = data_sesi['MA_3'].iloc[-1]
        ma_awal = data_sesi['MA_prev'].iloc
        growth_pct = ((ma_akhir / ma_awal) - 1) * 100
        
        total_sales = df[df['kode_produk'] == kode]['total_nilai'].sum()
        
        hasil_rising_star.append({
            'Kode Produk': kode,
            'Nama Produk': nama,
            'Growth %': round(growth_pct, 2),
            'Total Penjualan': total_sales
        })
        
    rising_star_df = pd.DataFrame(hasil_rising_star)
    print(f"Ditemukan {len(rising_star_df)} produk Rising Star.")

    # ==========================================
    # LANGKAH 3: POTENTIAL PACKAGING (APRIORI)
    # ==========================================
    # 1. Membentuk Matriks Keranjang
    basket = df.groupby(['nomor_struk', 'nama_produk'])['jumlah_terjual'].sum().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # 2. Algoritma Apriori & Association Rules
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # 3. Filtering: Mengandung Rising Star & Lift >= 2
    rising_star_products = set(rising_star_df['Nama Produk'])
    
    def contains_rising_star(itemset):
        return bool(itemset.intersection(rising_star_products))
        
    rules['has_rising_star'] = rules['antecedents'].apply(contains_rising_star) | rules['consequents'].apply(contains_rising_star)
    filtered_rules = rules[(rules['has_rising_star']) & (rules['lift'] >= 2)].copy()
    
    # 4. Sorting dan Formatting
    rules_sorted = filtered_rules.sort_values(by=['lift', 'support', 'confidence'], ascending=[False, False, False])
    
    rules_sorted['Jika Membeli'] = rules_sorted['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_sorted['Maka Membeli'] = rules_sorted['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_sorted['Jumlah Invoice'] = (rules_sorted['support'] * len(basket_sets)).astype(int)
    
    final_packaging_df = rules_sorted[['Jika Membeli', 'Maka Membeli', 'Jumlah Invoice', 'support', 'confidence', 'lift']]
    final_packaging_df = final_packaging_df.round({'support': 2, 'confidence': 2, 'lift': 2})

    # ==========================================
    # LANGKAH 4: EXPORT KE EXCEL
    # ==========================================
    with pd.ExcelWriter('retail_insight.xlsx', engine='openpyxl') as writer:
        rising_star_df.to_excel(writer, sheet_name='Rising Star', index=False)
        final_packaging_df.to_excel(writer, sheet_name='Potential Packaging', index=False)
    print("Berhasil mengekspor retail_insight.xlsx")

    # ==========================================
    # LANGKAH 5: VISUALISASI MATPLOTLIB
    # ==========================================
    # 1. Menyiapkan Data Top 3 vs Rising Star
    top_3_produk = df.groupby('nama_produk')['total_nilai'].sum().nlargest(3).index.tolist()
    produk_untuk_diplot = top_3_produk + list(rising_star_df['Nama Produk'])
    
    data_plot = daily_sales[daily_sales['nama_produk'].isin(produk_untuk_diplot)].copy()
    data_plot['MA_awal'] = data_plot.groupby('kode_produk')['MA_3'].transform('first')
    data_plot['Indeks_Base_100'] = (data_plot['MA_3'] / data_plot['MA_awal']) * 100

    # -- Grafik 1: Pertumbuhan Relatif (Base 100) --
    plt.figure(figsize=(12, 6))
    for nama in produk_untuk_diplot:
        subset = data_plot[data_plot['nama_produk'] == nama]
        
        # Style dasar (Gunakan spesifikasi warna presisi dari snippet_code_matplotlib.py Anda di sini)
        if nama in rising_star_products:
            plt.plot(subset['tgl_transaksi'], subset['Indeks_Base_100'], label=f'Rank 1: {nama}', color='#FFD700', linewidth=2.5, marker='o', markersize=4)
        else:
            plt.plot(subset['tgl_transaksi'], subset['Indeks_Base_100'], label=f'Top Sales: {nama}', color='#A9A9A9', linestyle='--', linewidth=1.5, marker='o', markersize=3)
            
    plt.ylabel('Indeks Pertumbuhan (Base 100)')
    plt.xlabel('Periode Tanggal')
    plt.title('ANALISIS PERTUMBUHAN RELATIF PRODUK RISING STAR\n(Dengan Benchmark Top 3 Total Penjualan)', fontweight='bold')
    plt.legend(title='Kategori Produk', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('rising_star_index.png', dpi=300)
    plt.close()
    print("Berhasil menyimpan rising_star_index.png")

    # -- Grafik 2: Nilai Penjualan Aktual --
    plt.figure(figsize=(12, 6))
    for nama in produk_untuk_diplot:
        subset = data_plot[data_plot['nama_produk'] == nama]
        
        if nama in rising_star_products:
            plt.plot(subset['tgl_transaksi'], subset['MA_3'], label=f'Rank 1: {nama}', color='#FFD700', linewidth=2.5, marker='o', markersize=4)
        else:
            plt.plot(subset['tgl_transaksi'], subset['MA_3'], label=f'Top Sales: {nama}', color='#A9A9A9', linestyle='--', linewidth=1.5, marker='o', markersize=3)
            
    plt.ylabel('Total Nilai Penjualan')
    plt.xlabel('Periode Tanggal')
    plt.title('ANALISIS NILAI PENJUALAN PRODUK RISING STAR\n(Nilai Penjualan Asli)', fontweight='bold')
    plt.legend(title='Kategori Produk', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('rising_star_actual.png', dpi=300)
    plt.close()
    print("Berhasil menyimpan rising_star_actual.png")
    
    print("Seluruh proses selesai! Anda siap untuk mengirimkan tugas ini.")

if __name__ == "__main__":
    main()