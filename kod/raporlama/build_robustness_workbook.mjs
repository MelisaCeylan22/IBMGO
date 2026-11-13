/** IBMGO deneylerinden sunum tablosu / XLSX oluşturur. */
import fs from "node:fs/promises";
const { SpreadsheetFile, Workbook } = await import(
  `${process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES}/@oai/artifact-tool/dist/artifact_tool.mjs`
);

const outDir = "sonuclar/dayaniklilik_analizi";
const data = JSON.parse(await fs.readFile(`${outDir}/analiz_verileri.json`, "utf8"));
const wb = Workbook.create();
const s = wb.worksheets.add("Sunum_Ozeti");
const z = wb.worksheets.add("Algoritma_Ozet");
const p = wb.worksheets.add("Eslesmis_Analiz");
const raw = wb.worksheets.add("Ham_Kosumlar");
const info = wb.worksheets.add("Deney_Protokolu");

const navy = "#1A2A44";
const blue = "#1169A0";
const orange = "#E57D24";
const gray = "#6E7D93";
const pale = "#F2F5FA";
const header = (sheet, address) => {
  const r = sheet.getRange(address);
  r.format = { fill: navy, font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center", verticalAlignment: "center", rowHeight: 28 };
};
const section = (sheet, address, color = blue) => {
  const r = sheet.getRange(address);
  r.format.font = { name: "Arial", size: 12, bold: true, color };
};
const sizes = [50, 100, 250, 500, 1000];
const find = (n, alg) => data.ozet.find(x => x.test_sayisi === n && x.algoritma === alg);

for (const sheet of [s, z, p, raw, info]) {
  sheet.showGridLines = false;
  sheet.getRange("A1:O1").format.rowHeight = 12;
  sheet.getRange("A2:O2").format.font = { name: "Arial", size: 14, color: navy, bold: true };
  sheet.getRange("A2:O45").format.verticalAlignment = "center";
}
s.tabColor = navy;
z.tabColor = "#577A9A";
info.tabColor = "#98A3B3";

s.getRange("A2").values = [["IBMGO | Bağımsız Koşumlarda Dayanıklılık"]];
s.getRange("A3").values = [[`3 sentetik veri kümesi (${data.veri_seedleri.join(", ")}) • her kümede 10 bağımsız başlangıç • sabit parametre`]];
s.getRange("A3:N3").format.font = { name: "Arial", size: 10, italic: true, color: "#526279" };
s.getRange("A5").values = [["Fitness karşılaştırması (düşük daha iyi)"]];
section(s, "A5");
s.getRange("A6:D11").values = [
  ["Test sayısı", "IBMGO", "GA", "Greedy"],
  ...sizes.map(n => [n, find(n, "IBMGO").fitness_ort, find(n, "GA").fitness_ort, find(n, "Greedy").fitness_ort]),
];
header(s, "A6:D6");
s.getRange("B7:D11").setNumberFormat("0.0000");
s.getRange("A7:D11").format.fill = "#F8FAFC";

s.getRange("A14").values = [["Koşum süresi tasarrufu (%, yüksek daha iyi)"]];
section(s, "A14");
s.getRange("A15:D20").values = [
  ["Test sayısı", "IBMGO", "GA", "Greedy"],
  ...sizes.map(n => [n, find(n, "IBMGO").sure_tasarrufu_ort,
                    find(n, "GA").sure_tasarrufu_ort, find(n, "Greedy").sure_tasarrufu_ort]),
];
header(s, "A15:D15");
s.getRange("B16:D20").setNumberFormat('0.00"%"');
s.getRange("A16:D20").format.fill = "#F8FAFC";

s.getRange("A23").values = [["Eşleşmiş IBMGO / GA koşumları"]];
section(s, "A23");
s.getRange("A24:F29").values = [
  ["Test", "IBMGO kazanır", "Eşit", "GA kazanır", "Ort. fitness farkı", "%95 aralık alt / üst"],
  ...data.eslesmis.map(r => [r.test_sayisi, r.bmgo_daha_iyi, r.eşit, r.ga_daha_iyi,
                              r.ortalama_fitness_farkı,
                              `[${r.bootstrap_alt_95.toFixed(5)}, ${r.bootstrap_üst_95.toFixed(5)}]`]),
];
header(s, "A24:F24");
s.getRange("E25:E29").setNumberFormat("+0.00000;-0.00000;0.00000");
s.getRange("A25:F29").format.fill = "#F8FAFC";

s.getRange("A32").values = [["Yorum"]];
section(s, "A32");
s.getRange("A33").values = [["Negatif fitness farkı IBMGO lehine; artı değer GA lehine."]];
s.getRange("A34").values = [["Greedy: veri kümesi başına tek deterministik sonuç; 30 tekrar değildir."]];
s.getRange("A35").values = [["Sentetik veri → gerçek üretim sistemi veya gerçek hata yakalama başarısı iddiası yok."]];
s.getRange("A36").values = [["%95 bootstrap aralığı açıklayıcıdır; yalnız 3 veri kümesiyle kesin genelleme yapılamaz."]];

const ch1 = s.charts.add("line", s.getRange("A6:D11"));
ch1.title = "Ortalama fitness";
ch1.setPosition("H5", "P19");
ch1.legend = { position: "top", textStyle: { typeface: "Arial" } };
ch1.xAxis = { axisType: "textAxis", textStyle: { typeface: "Arial", fontSize: 10 } };
ch1.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false,
              textStyle: { typeface: "Arial", fontSize: 10 } };
for (let i=0; i<ch1.series.items.length; i++) {
  ch1.series.items[i].line = { fill: [blue, orange, gray][i], style: "solid", width: 2 };
}
const ch2 = s.charts.add("line", s.getRange("A15:D20"));
ch2.title = "Ortalama süre tasarrufu (%)";
ch2.setPosition("H21", "P36");
ch2.legend = { position: "top", textStyle: { typeface: "Arial" } };
ch2.xAxis = { axisType: "textAxis", textStyle: { typeface: "Arial", fontSize: 10 } };
ch2.yAxis = { numberFormatCode: '0"%"', numberFormatSourceLinked: false,
              textStyle: { typeface: "Arial", fontSize: 10 } };
for (let i=0; i<ch2.series.items.length; i++) {
  ch2.series.items[i].line = { fill: [blue, orange, gray][i], style: "solid", width: 2 };
}

s.getRange("A:A").format.columnWidth = 17;
s.getRange("B:D").format.columnWidth = 19;
s.getRange("E:E").format.columnWidth = 22;
s.getRange("F:F").format.columnWidth = 32;
s.getRange("G:G").format.columnWidth = 4;

function addTable(sheet, title, rows, mapping, widths) {
  sheet.getRange("A2").values = [[title]];
  const values = [mapping.map(([h]) => h), ...rows.map(x => mapping.map(([,key]) => x[key] ?? ""))];
  sheet.getRangeByIndexes(3, 0, values.length, mapping.length).values = values;
  const colEnd = String.fromCharCode(64 + mapping.length);
  header(sheet, `A4:${colEnd}4`);
  sheet.freezePanes.freezeRows(4);
  for (let col=0; col<mapping.length; col++) {
    const letter = String.fromCharCode(65 + col);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widths[col] ?? 17;
  }
  const rowEnd = 3 + values.length;
  sheet.getRange(`A5:${colEnd}${rowEnd}`).format.font = { name: "Arial", size: 10, color: "#223149" };
  sheet.getRange(`A5:${colEnd}${rowEnd}`).format.rowHeight = 20;
}

addTable(z, "Algoritma × Test Boyutu | Ortalamalar ve Standart Sapmalar", data.ozet, [
  ["Test sayısı", "test_sayisi"], ["Algoritma", "algoritma"],
  ["Kayıt sayısı", "kayıt"], ["Veri kümesi", "veri_kümesi"],
  ["Fitness ort.", "fitness_ort"], ["Fitness SS", "fitness_ss"],
  ["Tasarruf ort. (%)", "sure_tasarrufu_ort"], ["Tasarruf SS", "sure_tasarrufu_ss"],
  ["Test adet ort.", "secilen_test_ort"], ["Algoritma sn ort.", "algoritma_suresi_ort"],
], [15, 18, 15, 16, 16, 14, 18, 15, 17, 20]);
z.getRange("E5:F19").setNumberFormat("0.000000");
z.getRange("G5:H19").setNumberFormat("0.00");
z.getRange("I5:J19").setNumberFormat("0.000");

addTable(p, "IBMGO − GA | Aynı Veri ve Algoritma Başlangıcı", data.eslesmis, [
  ["Test sayısı", "test_sayisi"], ["Çift", "çift"],
  ["IBMGO kazanır", "bmgo_daha_iyi"], ["Eşit", "eşit"],
  ["GA kazanır", "ga_daha_iyi"], ["IBMGO kazanma %", "bmgo_kazanma_yüzdesi"],
  ["Ort. fitness farkı", "ortalama_fitness_farkı"],
  ["%95 alt", "bootstrap_alt_95"], ["%95 üst", "bootstrap_üst_95"],
], [15, 12, 18, 12, 16, 21, 22, 16, 16]);
p.getRange("G5:I9").setNumberFormat("0.000000");

addTable(raw, "Ham deney sonuçları | Dosyaların birleştirilmiş hali", data.ham_kosumlar, [
  ["Test", "test_sayisi"], ["Veri seed", "veri_seed"],
  ["Algoritma", "algoritma"], ["Opt. seed", "algoritma_seed"],
  ["Fitness", "fitness"], ["Seçili test", "secilen_test"],
  ["Tasarruf (%)", "sure_tasarrufu"], ["Kapsam (%)", "genel_kapsam"],
  ["Kritik eksik", "kritik_eksik"], ["Ön koşul ihlal", "kopuk_on_kosul"],
  ["Aday değerlendirme", "aday_degerlendirmesi"], ["Algoritma süresi", "algoritma_suresi"],
], [12, 15, 28, 15, 16, 16, 17, 16, 15, 19, 22, 20]);
raw.getRange(`E5:E${4 + data.ham_kosumlar.length}`).setNumberFormat("0.000000");

info.getRange("A2").values = [["Deney protokolü ve doğru yorumlama"]];
info.getRange("A4:B14").values = [
  ["Amaç", "IBMGO'nin istikrarını yeni rastgele başlangıçlarda ölçmek"],
  ["Test boyutları", sizes.join(" / ")],
  ["Bağımsız veri seed", data.veri_seedleri.join(" / ")],
  ["Optimizer başlangıçları", `42–${41+data.tekrar} (${data.tekrar} tekrar)`],
  ["Fitness bütçesi", "IBMGO ve GA için 1212 aday / koşum"],
  ["Kısıt 1", "Kritik gereksinim kapsamı %100"],
  ["Kısıt 2", "Genel kapsam en az %90"],
  ["Kısıt 3", "Ön koşul bağımlılığı ihlali = 0"],
  ["Yöntem", "Fitness farkları veri+optimizer seed eşleştirmesiyle hesaplandı"],
  ["Güven aralığı", "Veri kümesi ve optimizer seed düzeyinde hiyerarşik bootstrap"],
  ["Sınırlama", "Yalnız 3 sentetik veri kümesi; sonuçlar genellenemez"],
];
info.getRange("A4:A14").format.font = { name:"Arial", bold:true, color:navy, size:10 };
info.getRange("A:A").format.columnWidth = 28;
info.getRange("B:B").format.columnWidth = 86;
info.getRange("B4:B14").format.font = { name:"Arial", color:"#364760", size:10 };
info.getRange("A16").values = [["Not: Gerçek hata matrisi olmadığı için APFD ölçülememiştir."]];
info.getRange("A17").values = [["Greedy, deterministik tek-geçiş yaklaşımıdır; fitness aday bütçeleri eşit sayılmamalıdır."]];

// Kullanıcının paylaştığı yaklaşık önce/sonra süreleri. Mevcut deney
// tabloları korunur; bu karşılaştırma özetin altındaki boş alana eklenir.
s.getRange("A40").values = [["IBMGO öncesi ve sonrası toplam test süresi"]];
section(s, "A40");
s.getRange("A41:E46").values = [
  ["Test sayısı", "Tüm testler (dk)", "IBMGO (dk)", "Kazanç (dk)", "Süre tasarrufu"],
  [50, 31.2, 17.6, null, 0.4354],
  [100, 64.5, 30.7, null, 0.5238],
  [250, 2.73 * 60, 43.0, null, 0.7369],
  [500, 5.54 * 60, 1.07 * 60, null, 0.8063],
  [1000, 11.16 * 60, 1.51 * 60, null, 0.8644],
];
header(s, "A41:E41");
s.getRange("A42:E46").format.fill = "#F8FAFC";
s.getRange("A42:E46").format.font = { name:"Arial", size:10, color:"#223149" };
s.getRange("A42:E46").format.rowHeight = 23;
s.getRange("B42:D46").setNumberFormat("0.0");
s.getRange("E42:E46").setNumberFormat("0.00%");
s.getRange("D42:D46").formulas = sizes.map((_, i) => [`=B${42+i}-C${42+i}`]);
s.getRange("A48").values = [["Sentetik senaryoların tahmini toplam test süreleridir. Gerçek CI/CD yürütme ölçümü değildir."]];
s.getRange("A49").values = [["Kaynak: Paylaşılan önce/sonra tablosu. Süreler yuvarlatılmıştır; tasarruf yüzdeleri bildirildiği gibi korunmuştur."]];
s.getRange("A50").values = [["Saat cinsindeki süreler grafikte karşılaştırılabilmesi için dakikaya çevrilmiştir (1 saat = 60 dk)."]];
s.getRange("A48:F50").format.font = { name:"Arial", size:10, italic:true, color:"#526279" };
const durationChart = s.charts.add("bar", s.getRange("A41:C46"));
durationChart.title = "Toplam test süresi (dakika)";
durationChart.titleTextStyle.typeface = "Arial";
durationChart.titleTextStyle.fontSize = 14;
durationChart.setPosition("H40", "P57");
durationChart.legend = { position:"top", textStyle:{ typeface:"Arial", fontSize:11 } };
durationChart.xAxis = { axisType:"textAxis", textStyle:{ typeface:"Arial", fontSize:11 } };
durationChart.yAxis = { numberFormatCode:"0", numberFormatSourceLinked:false,
  textStyle:{ typeface:"Arial", fontSize:11 } };
durationChart.series.items[0].fill = gray;
durationChart.series.items[1].fill = blue;

wb.recalculate();
await fs.mkdir(outDir, { recursive:true });
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(`${outDir}/IBMGO_Dayaniklilik_Sunum_Tablolari.xlsx`);
const inspect = await wb.inspect({ kind: "sheet", include: "id,name", maxChars: 1600 });
console.log(JSON.stringify(inspect).slice(0, 1500));
try {
  const png = await wb.render({ sheetName: "Sunum_Ozeti", range: "A1:P57", scale: 1, format: "png" });
  await fs.writeFile(`${outDir}/04_excel_sunum_onizleme.png`, new Uint8Array(await png.arrayBuffer()));
  console.log("Excel önizlemesi hazır.");
  const comparison = await wb.render({ sheetName:"Sunum_Ozeti", range:"A39:P57", scale:1, format:"png" });
  await fs.writeFile(`${outDir}/05_sure_karsilastirma.png`, new Uint8Array(await comparison.arrayBuffer()));
} catch (error) {
  console.warn("Önizleme alınamadı, Excel dosyası oluşturuldu:", error.message);
}
console.log(`Excel: ${outDir}/IBMGO_Dayaniklilik_Sunum_Tablolari.xlsx`);
