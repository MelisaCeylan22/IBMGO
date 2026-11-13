"""Sunum notebook'unun etkileşimli kumandaları. Hesaplama sunum_deneyi ile ortaktır."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output

from kod.deneyler.sunum_deneyi import (Settings, PresentationExperiment, summarize,
                                       result_figures, convergence_figures, selected_table)


class PresentationPanel:
    def __init__(self, experiment=None):
        import ipywidgets as w
        self.w=w
        self.experiment=experiment or PresentationExperiment()
        self.controls={}
        self.updating=False
        self.inputs_dirty=True
        self.status=w.HTML()
        self.results=w.Output()
        self.data_output=w.Output()
        self.budget_output=w.Output()
        self.progress=w.IntProgress(min=0,max=1,description='Koşum')
        self.run_button=w.Button(description='Deneyi çalıştır / Sonuçları güncelle',button_style='success',layout=w.Layout(width='340px'))
        self.archive_button=w.Button(description='Kayıtlı ana deneyi göster',layout=w.Layout(width='230px'))
        self.mode=w.Dropdown(options=['Hızlı canlı demo','Tam dayanıklılık'],description='Hazır mod')
        s=self.experiment.settings
        for key,label in [('sizes','Test boyutları'),('data_seeds','Veri tohumları'),('algorithm_seeds','Algoritma tohumları')]:
            self.controls[key]=w.Text(value=','.join(map(str,getattr(s,key))),description=label,style={'description_width':'170px'},layout=w.Layout(width='550px'))
        for key,label in [('min_coverage','Asgari genel kapsam'),('w_time','Süre ağırlığı'),('w_count','Test sayısı ağırlığı'),('w_redundancy','Tekrar ağırlığı'),('w_rare','Nadir kayıp ağırlığı')]:
            self.controls[key]=w.FloatText(value=getattr(s,key),description=label,style={'description_width':'170px'})
        for key,label in [('population','Popülasyon'),('iterations','IBMGO iterasyon'),('local_trials','Yerel aday / iterasyon')]:
            self.controls[key]=w.IntText(value=getattr(s,key),description=label,style={'description_width':'170px'})
        self.controls['adaptive_policy']=w.Checkbox(value=s.adaptive_policy,description='Boyuta duyarlı politika')
        self.controls['advanced']=w.Textarea(value='{}',description='İleri ayarlar JSON',style={'description_width':'170px'},layout=w.Layout(width='650px'))
        self.run_button.on_click(self.run)
        self.archive_button.on_click(self.archive)
        self.mode.observe(self.preset,names='value')
        for control in self.controls.values(): control.observe(self.changed,names='value')
        self.settings_view=w.VBox([self.mode,*self.controls.values(),self.budget_output,
            w.HBox([self.run_button,self.archive_button]),self.progress,self.status])
        self.pool_choice=w.Dropdown(description='Veri kopyası',layout=w.Layout(width='550px'))
        self.test_index=w.BoundedIntText(value=0,min=0,max=49,description='Test indeksi')
        self.feature_index=w.BoundedIntText(value=0,min=0,max=45,description='Hedef indeksi')
        self.duration=w.FloatText(value=1,description='Süre (sn)')
        self.is_critical=w.Checkbox(description='Kritik hedef')
        self.covered=w.Checkbox(description='Bu test bu hedefi kapsar')
        self.apply_button=w.Button(description='Veri değişikliğini uygula',layout=w.Layout(width='240px'))
        self.reset_button=w.Button(description='Orijinal veriye dön')
        self.apply_button.on_click(self.apply_edit);self.reset_button.on_click(self.reset)
        for control in [self.pool_choice,self.test_index,self.feature_index]:control.observe(self.show_data,names='value')
        self.editor=w.VBox([self.pool_choice,w.HBox([self.test_index,self.feature_index]),w.HBox([self.duration,self.is_critical,self.covered]),w.HBox([self.apply_button,self.reset_button]),self.data_output])
        self.size_filter=w.Dropdown(description='Grafik boyutu')
        self.run_filter=w.Dropdown(description='Koşum',layout=w.Layout(width='650px'))
        self.raw_toggle=w.Checkbox(value=False,description='Ham koşum tablosunu göster')
        self.size_filter.observe(self.filter_changed,names='value')
        self.run_filter.observe(self.render,names='value');self.raw_toggle.observe(self.render,names='value')
        self.result_view=w.VBox([w.HBox([self.size_filter,self.raw_toggle]),self.run_filter,self.results])
        self.changed()

    def read_settings(self):
        values={}
        for key,control in self.controls.items():
            value=control.value
            if key in ['sizes','data_seeds','algorithm_seeds']:
                try:value=[int(v.strip()) for v in value.split(',') if v.strip()]
                except ValueError:raise ValueError('Boyut ve tohum alanlarında virgülle ayrılmış tam sayılar kullanın.')
            if key=='advanced':
                try:value=json.loads(value)
                except json.JSONDecodeError:raise ValueError('İleri ayarlar geçerli bir JSON nesnesi olmalı.')
            values[key]=value
        return Settings(mode=self.mode.value,**values).validate()

    def preset(self,change):
        config=Settings.full() if self.mode.value=='Tam dayanıklılık' else Settings()
        self.updating=True
        for key,control in self.controls.items():
            value=getattr(config,key)
            control.value=','.join(map(str,value)) if isinstance(value,list) else json.dumps(value) if key=='advanced' else value
        self.updating=False;self.changed()

    def changed(self,change=None):
        if self.updating:return
        self.inputs_dirty=True
        try:
            self.experiment.settings=self.read_settings()
            self.updating=True
            pairs=[(f'{n} test · veri {s}',(n,s)) for n in self.experiment.settings.sizes for s in self.experiment.settings.data_seeds]
            old=self.pool_choice.value
            self.pool_choice.options=pairs
            self.pool_choice.value=old if old in [p[1] for p in pairs] else pairs[0][1]
            self.updating=False
            with self.budget_output:
                clear_output(wait=True)
                display(pd.DataFrame([dict(test=n,**self.experiment.settings.budget(n)) for n in self.experiment.settings.sizes]))
                print('Bütçeler planlanandır. Fark = GA − IBMGO. Greedy eşit bütçeli değildir.')
            self.status.value='<b>Güncel değil.</b> Ayar/veri değişikliğini hesaplamak için Deneyi çalıştır düğmesine basın.'
            self.run_button.disabled=False
            self.show_data()
        except (ValueError,TypeError) as exc:
            self.updating=False
            self.status.value='<b>Geçersiz ayar.</b> '+str(exc)
            self.run_button.disabled=True
        self.render()

    def show_data(self,change=None):
        if self.updating or self.pool_choice.value is None:return
        n,s=self.pool_choice.value;pool=self.experiment.pool(n,s)
        self.updating=True
        self.test_index.max=n-1;self.feature_index.max=pool['coverage'].shape[1]-1
        i,j=self.test_index.value,self.feature_index.value
        self.duration.value=float(pool['durations'][i])
        self.is_critical.value=j in pool['critical'];self.covered.value=bool(pool['coverage'][i,j])
        self.updating=False
        with self.data_output:
            clear_output(wait=True)
            case=pool['metadata']['test_cases'][i]
            print(f"{case['id']} · {pool['names'][i]}\nBeklenen sonuç · {case['expected_result']}\nÖn koşul indeksleri · {np.flatnonzero(pool['dependencies'][:,i]).tolist()}")
            print('Hedef · '+pool['features'][j])
            # En fazla 8×8 hücre. Seçilen satır/sütun da görünümde kalır.
            rows=list(range(max(0,i-3),min(n,i+5)));cols=list(range(max(0,j-3),min(len(pool['features']),j+5)))
            display(pd.DataFrame(pool['coverage'][np.ix_(rows,cols)],index=rows,columns=[str(k)+(' ★' if k in pool['critical'] else '') for k in cols]))
            print('İndeksler sıfırdan başlar. Değişiklikler yalnız seçilen havuzun sunum kopyasına uygulanır.')

    def apply_edit(self,_=None):
        n,s=self.pool_choice.value;pool=self.experiment.pool(n,s)
        i,j=self.test_index.value,self.feature_index.value
        pool['durations'][i]=self.duration.value
        pool['coverage'][i,j]=int(self.covered.value)
        critical=set(pool['critical'].tolist())
        critical.add(j) if self.is_critical.value else critical.discard(j)
        pool['critical']=np.asarray(sorted(critical),dtype=int)
        self.experiment.set_pool(n,s,pool)
        self.changed()

    def reset(self,_=None):
        self.experiment.reset_pool(*self.pool_choice.value);self.changed()

    def archive(self,_=None):
        try:
            self.experiment.load_archive();self.update_filters()
            self.status.value='<b>Kayıtlı deney.</b> Ana arşiv gösteriliyor. Paneldeki canlı ayarlar bu sonuçlara uygulanmadı.'
        except Exception as exc:self.status.value='Arşiv yüklenemedi · '+str(exc)

    def run(self,_=None):
        self.run_button.disabled=True
        self.archive_button.disabled=True
        for c in self.controls.values():c.disabled=True
        for c in [self.mode,self.apply_button,self.reset_button]:c.disabled=True
        try:
            self.experiment.settings=self.read_settings()
            def progress(done,total,key):
                self.progress.max=total;self.progress.value=done
                self.status.value=f'<b>Hesaplanıyor</b> · {done}/{total} koşum tamamlandı · {key}'
            self.status.value='Veri ve kısıtlar doğrulanıyor…'
            self.experiment.run(progress)
            self.inputs_dirty=False
            self.update_filters()
            self.status.value='<b>Yeni çalıştırma tamamlandı.</b> '+self.experiment.result['paths'][0]
        except Exception as exc:
            self.status.value='<b>Çalıştırma tamamlanamadı. Önceki sonuç korunuyor.</b> '+str(exc)
            self.render()
        finally:
            self.run_button.disabled=False;self.archive_button.disabled=False
            for c in self.controls.values():c.disabled=False
            for c in [self.mode,self.apply_button,self.reset_button]:c.disabled=False

    def update_filters(self):
        self.updating=True
        old=self.size_filter.value
        self.size_filter.options=sorted(self.experiment.result['rows']['size'].unique().tolist())
        self.size_filter.value=old if old in self.size_filter.options else self.size_filter.options[0]
        self.updating=False;self.filter_changed()

    def filter_changed(self,change=None):
        if self.updating:return
        result=self.experiment.result
        if result is None:return
        self.updating=True
        old=self.run_filter.value
        self.run_filter.options=[k for k,r in result['runs'].items() if r['size']==self.size_filter.value and r['method']!='Greedy']
        self.run_filter.value=old if old in self.run_filter.options else self.run_filter.options[0] if self.run_filter.options else None
        self.updating=False;self.render()

    def render(self,change=None):
        if self.updating:return
        result=self.experiment.result
        if result is None:return
        with self.results:
            clear_output(wait=True)
            display(HTML('<h3>'+result['source']+('</h3>' if not (self.inputs_dirty or self.experiment.stale) else ' · Panel ayarları için güncel değil</h3>')))
            print(result['note'])
            display(pd.Series(result['settings'],name='Bu sonuçların ayarları').to_frame())
            print('Kaynak · '+' | '.join(result['paths']))
            size=self.size_filter.value
            if size is None:return
            frame=result['rows'];subset=frame[frame['size']==size]
            print('Ortalama özeti · Standart sapma örnek standart sapmasıdır. Tek koşumda hesaplanmaz.')
            summary=summarize(subset)
            display(summary[['method','kosum_sayisi','veri_kumesi','algoritma_tohumu']].rename(columns={
                'method':'Yöntem','kosum_sayisi':'Koşum sayısı','veri_kumesi':'Veri kümesi','algoritma_tohumu':'Farklı algoritma tohumu'}))
            from kod.deneyler.sunum_deneyi import METRICS, LABELS
            compact={}
            for _,row in summary.iterrows():
                compact[row['method']]=[
                    'Kayıt yok' if pd.isna(row[metric+'_mean']) else
                    f"{row[metric+'_mean']:.5g}" +
                    (f" ± {row[metric+'_std']:.3g}" if pd.notna(row[metric+'_std']) else ' (tek koşum)')
                    for metric in METRICS]
            display(pd.DataFrame(compact,index=LABELS))
            if self.raw_toggle.value:
                print('Ham koşumlar · en fazla ilk 100 kayıt');display(subset.head(100))
            fig=result_figures(result,size);display(fig);plt.close(fig)
            key=self.run_filter.value
            if key is None:
                print('Arşiv CSV’lerinde yakınsama geçmişi ve seçili testler yok. Bunları göstermek için yeni çalıştırma başlatın.')
            else:
                r=result['runs'][key]
                labels={'initial_fitness':'Başlangıç fitness', 'final_fitness':'Nihai en iyi fitness',
                        'steps':'Toplam iterasyon / nesil', 'evaluations':'Gerçek amaç çağrısı',
                        'first_final_step':'Nihai en iyiye ilk erişim adımı',
                        'first_final_evaluation':'Nihai en iyiye ilk erişim çağrısı',
                        'last_improvement_step':'Son iyileşme adımı',
                        'last_improvement_evaluation':'Son iyileşme çağrısı',
                        'last_improvement_phase':'Son iyileşmenin aşaması', 'arama_sn':'Arama süresi (sn)'}
                display(pd.Series({labels[k]:v for k,v in (r['trace']|{'arama_sn':r['arama_sn']}).items() if k!='events'},name='Seçilmiş koşum').to_frame())
                fig=convergence_figures(result,key);display(fig);plt.close(fig)
                print('Nihai en iyiye ilk erişim, bu koşumun sonuna kadar bulunan en iyi puana ilk erişimdir. Tolerans rtol=1e-10, atol=1e-12. Küresel optimum iddiası içermez.')
                print('Seçilen testler · Bağımlılıkları koruyan öncelik sırası · İlk 20 test')
                display(selected_table(result,key))
