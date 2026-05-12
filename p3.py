import pandas as pd
import numpy as np
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# load sama rapihin dataset 1
df1 = pd.read_csv('ai_job_replacement_cleaned.csv')
# samain nama industri biar ga pusing, tech jadi technology, soalnya pas direplace langsung malah gabisa for some reason
df1['industry'] = df1['industry'].str.lower().replace('tech', 'technology').str.title()

# load trus bersihin dataset 2 (buka bungkus kutipnya)
parsed_data = []
with open('ai_impact_jobs_cleaned.csv', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            # buka bungkus kutip luar trus ganti kutip double jadi satu
            line = line[1:-1].replace('""', '"')
        parsed_data.append(line)

df2 = pd.read_csv(io.StringIO('\n'.join(parsed_data)))
df2['industry'] = df2['industry'].str.lower().replace('tech', 'technology').str.title()

# filter biar fokus ke 5 industri target aja, biar ga numpuk 
target_industries = ['Technology', 'Finance', 'Manufacturing', 'Healthcare', 'Retail']
df1 = df1[df1['industry'].isin(target_industries)]
df2 = df2[df2['industry'].isin(target_industries)]

# itung rata2 variabel per industri ama ambil data urgensi ama tekanan transisi dari dataset 1
agg1 = df1.groupby('industry')[['reskilling_urgency_score', 'skill_transition_pressure']].mean().reset_index()

# ambil data intensitas ai sama risiko otomatisasi dari dataset 2
agg2 = df2.groupby('industry')[['ai_intensity_score', 'automation_risk_score']].mean().reset_index()
# jadiin persen biar match, soalnya beda satuannya ituan variabelnya
agg2['automation_risk_percent_ds2'] = agg2['automation_risk_score'] * 100

# gabungin jadi satu tabel pake kolom industri buat kuncinya
merged = pd.merge(agg1, agg2, on='industry')

# kmeans clustering pake 4 variabel gabungan tadi
features_to_cluster = [
    'reskilling_urgency_score', 
    'skill_transition_pressure', 
    'ai_intensity_score', 
    'automation_risk_percent_ds2'
]

X = merged[features_to_cluster]
# standarisasi biar skalanya adil pas dihitung k-means, terimakasih youtube codebasics
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# bikin 3 klaster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged['cluster'] = kmeans.fit_predict(X_scaled)

# kasih label prioritas (yang skor urgensinya paling gede = tinggi)
cluster_means = merged.groupby('cluster')['reskilling_urgency_score'].mean().sort_values()
mapping = {cluster_means.index[0]: 'Rendah', cluster_means.index[1]: 'Menengah', cluster_means.index[2]: 'Tinggi'}
merged['Priority'] = merged['cluster'].map(mapping)

# urutin biar grafiknya cakep dari bawah ke atas
merged = merged.sort_values('reskilling_urgency_score', ascending=True)

# bikin grafik batang horizontal
plt.figure(figsize=(10, 6))
colors = {'Tinggi': '#d62728', 'Menengah': '#ff7f0e', 'Rendah': '#ffcc00'}

bars = plt.barh(
    merged['industry'],
    merged['reskilling_urgency_score'],
    color=[colors[p] for p in merged['Priority']],
    edgecolor='black'
)

# taro angka skor di ujung batang biar jelas, kayaknya si
for bar in bars:
    plt.text(
        bar.get_width() + 0.5, 
        bar.get_y() + bar.get_height()/2, 
        f'{bar.get_width():.1f}', 
        va='center', 
        ha='left', 
        fontsize=10
    )

plt.title('Skor Urgensi Pelatihan Ulang per Industri\n(Gabungan Dataset 1 & 2)', pad=15, fontweight='bold')
plt.xlabel('Skor Urgensi Pelatihan Ulang')
plt.ylabel('Industri')
plt.xlim(0, max(merged['reskilling_urgency_score']) + 5)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# bikin legend sendiri biar orang paham maksud warnanya
legend_elements = [Patch(facecolor=colors[k], edgecolor='black', label=f'Prioritas {k}') for k in ['Tinggi', 'Menengah', 'Rendah']]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
# YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
plt.savefig('bismillah ga error kodenya, youtube dataquest tolong kami, yes grafik bener yessss.png', dpi=300)