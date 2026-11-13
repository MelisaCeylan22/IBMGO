"""Notebook ve widgetsız kullanım için ortak, kayıtlı sunum deneyi."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kod.algoritmalar.bmgo import binary_mountain_gazelle_optimizer, dimension_adaptive_settings
from kod.algoritmalar.baselines import genetic_algorithm_optimizer, greedy_regression_selector
from kod.algoritmalar.fitness import get_suite_metrics, prioritize_tests_and_calculate_apfd
from kod.veri.data_generator import generate_telecom_data

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['fitness', 'tasarruf_yuzde', 'tam_paket_sn', 'secili_paket_sn',
           'secilen_test', 'arama_sn', 'kritik_kapsam_yuzde', 'genel_kapsam_yuzde', 'on_kosul_ihlali']
LABELS = ['Fitness ↓', 'Süre tasarrufu (%)', 'Tam paket tahmini (sn)',
          'Seçili paket tahmini (sn)', 'Seçilen test', 'Arama süresi (sn)',
          'Kritik kapsam (%)', 'Genel kapsam (%)', 'Ön koşul ihlali']


@dataclass
class Settings:
    mode: str = 'Hızlı canlı demo'
    sizes: list = field(default_factory=lambda: [50])
    data_seeds: list = field(default_factory=lambda: [83])
    algorithm_seeds: list = field(default_factory=lambda: [42])
    min_coverage: float = .9
    w_time: float = .4
    w_count: float = .4
    w_redundancy: float = .2
    w_rare: float = 0.
    population: int = 12
    iterations: int = 20
    local_trials: int = 12
    adaptive_policy: bool = True
    advanced: dict = field(default_factory=dict)

    @classmethod
    def full(cls):
        return cls(mode='Tam dayanıklılık', sizes=[50, 100, 250, 500, 1000],
                   data_seeds=[83, 97, 109], algorithm_seeds=list(range(42, 52)))

    def options(self, size):
        options = dimension_adaptive_settings(size) if self.adaptive_policy else {}
        options.update(local_search_trials=self.local_trials)
        options.update(self.advanced)
        return options

    def budget(self, size):
        options = self.options(size)
        planned = self.population + self.iterations * (
            4*self.population + self.local_trials + options.get('spiral_search_trials', 0))
        generations = max(1, int(np.floor((planned-self.population)/self.population + .5)))
        return dict(ibmgo=planned, ga=self.population*(1+generations),
                    generations=generations, difference=self.population*(1+generations)-planned)

    def validate(self):
        for values, label, minimum in [(self.sizes, 'Test boyutları', 50),
                                        (self.data_seeds, 'Veri tohumları', 0),
                                        (self.algorithm_seeds, 'Algoritma tohumları', 0)]:
            if not isinstance(values, list) or not values or any(type(v) is not int or v < minimum for v in values) or len(set(values)) != len(values):
                raise ValueError(f'{label} farklı tam sayılardan oluşan bir liste olmalı, alt sınır {minimum}.')
        for value, name, minimum in [(self.population,'Popülasyon',2), (self.iterations,'İterasyon',1), (self.local_trials,'Yerel deneme',0)]:
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} en az {minimum} olan tam sayı olmalı.')
        weights = [self.w_time,self.w_count,self.w_redundancy,self.w_rare]
        if not np.all(np.isfinite(weights)) or min(weights)<0 or sum(weights)<=0:
            raise ValueError('Ağırlıklar sonlu ve negatif olmayan sayılar olmalı, toplamları pozitif olmalı.')
        if not np.isfinite(self.min_coverage) or not 0<self.min_coverage<=1:
            raise ValueError('Genel kapsam 0 ile 1 arasında, sıfırdan büyük olmalı.')
        if type(self.adaptive_policy) is not bool:
            raise ValueError('Boyuta duyarlı politika True veya False olmalı.')
        allowed = {'adaptive_binary','diverse_initialization','reference_mgo_formulas',
                   'chaotic_initialization','nonlinear_control','spiral_search_trials',
                   'binary_transfer','reference_mgo_probability','flip_rate_start',
                   'flip_rate_end','local_search_strategy'}
        if not isinstance(self.advanced,dict) or not set(self.advanced)<=allowed:
            raise ValueError('İleri ayarlarda desteklenmeyen anahtar var. Yerel denemeyi ana panelden değiştirin.')
        for key, value in self.advanced.items():
            if key in {'adaptive_binary','diverse_initialization','reference_mgo_formulas','chaotic_initialization','nonlinear_control'} and type(value) is not bool:
                raise ValueError(f'{key} True veya False olmalı.')
        opt=self.options(self.sizes[0])
        spiral=opt.get('spiral_search_trials',0)
        if type(spiral) is not int or spiral<0:
            raise ValueError('Spiral deneme negatif olmayan tam sayı olmalı.')
        if opt.get('binary_transfer','v_flip') not in {'v_flip','s_assign'} or opt.get('local_search_strategy','swap') not in {'swap','large'}:
            raise ValueError('İkili dönüşüm veya komşuluk seçeneği geçersiz.')
        probability=opt.get('reference_mgo_probability',1.)
        start,end=opt.get('flip_rate_start',.04),opt.get('flip_rate_end',.02)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in [probability, start, end]):
            raise ValueError('Olasılık ve bit değişim oranları sonlu sayılar olmalı.')
        if not 0<=probability<=1 or not 0<end<=start<=1:
            raise ValueError('Olasılık [0,1], bit değişimi 0 < bitiş ≤ başlangıç ≤ 1 olmalı.')
        return self


def validate_pool(pool, minimum):
    c,d,dep,critical=pool['coverage'],pool['durations'],pool['dependencies'],pool['critical']
    if c.ndim!=2 or not c.size or not np.isin(c,[0,1]).all():
        raise ValueError('Kapsam matrisi boş olamaz ve yalnız 0/1 içermeli.')
    n,f=c.shape
    if d.shape!=(n,) or not np.isfinite(d).all() or np.any(d<=0):
        raise ValueError('Her testin süresi sonlu ve sıfırdan büyük olmalı.')
    if dep.shape!=(n,n) or not np.isin(dep,[0,1]).all() or np.any(np.tril(dep)):
        raise ValueError('Ön koşullar sıralı ve döngüsüz olmalı. Ebeveyn indeksi çocuktan küçük olmalı.')
    if critical.ndim!=1 or not np.issubdtype(critical.dtype,np.integer) or np.any(critical<0) or np.any(critical>=f):
        raise ValueError('Kritik hedef indeksleri geçersiz.')
    available=c.any(axis=0)
    missing=critical[~available[critical]]
    if len(missing):
        raise ValueError('Kapsanması olanaksız kritik hedefler: '+', '.join(pool['features'][i] for i in missing))
    if available.sum()<np.ceil(minimum*f-1e-12):
        raise ValueError(f'Genel kapsam olanaksız. En fazla {available.sum()}/{f} hedef kapsanabilir.')


def make_pool(size, seed):
    c,d,names,features,dep,meta=generate_telecom_data(size,seed=seed,scenario_profile='expanded',include_metadata=True)
    return dict(coverage=c.copy(),durations=d.astype(float),dependencies=dep.copy(),
                critical=np.asarray(meta['critical_feature_indices'],dtype=int),
                names=names,features=features,metadata=meta)


def digest_pool(pool):
    h=hashlib.sha256()
    for key in ['coverage','durations','dependencies','critical']:
        a=np.ascontiguousarray(pool[key]);h.update(key.encode());h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


class PresentationExperiment:
    def __init__(self, settings=None, root=ROOT):
        self.settings=settings or Settings()
        self.root=Path(root)
        self.edits={}
        self.result=None
        self.result_fingerprint=None
        self.last_error=None

    def pool(self,size,seed):
        # Her çağrı bağımsız kopya döndürür. Kaynak üreticiye yazılmaz.
        return deepcopy(self.edits.get((size,seed))) if (size,seed) in self.edits else make_pool(size,seed)

    def set_pool(self,size,seed,pool):
        self.edits[(size,seed)]=deepcopy(pool)

    def reset_pool(self,size,seed):
        self.edits.pop((size,seed),None)

    def fingerprint(self):
        self.settings.validate()
        content=dict(settings=asdict(self.settings),pools={f'{n}_{s}':digest_pool(self.pool(n,s)) for n in self.settings.sizes for s in self.settings.data_seeds})
        return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()

    @property
    def stale(self):
        if self.result is None or self.result['source']=='Kayıtlı deney':
            return True
        try:
            return self.fingerprint()!=self.result_fingerprint
        except (ValueError,TypeError):
            return True

    def load_archive(self):
        from kod.deneyler.robustness_analysis import _read_csv, _validate_design
        paths=[self.root/'sonuclar'/f'dayaniklilik_{s}'/f'ablasyon_veri{s}.csv' for s in [83,97,109]]
        rows=_read_csv(paths)
        _validate_design(rows,10,(50,100,250,500,1000))
        records=[]
        for r in rows:
            records.append(dict(size=r['test_sayisi'],data_seed=r['veri_seed'],algorithm_seed=r['algoritma_seed'],
                method={'BMGO_boyuta_adaptif':'IBMGO'}.get(r['algoritma'],r['algoritma']),
                fitness=r['fitness'],tasarruf_yuzde=r['sure_tasarrufu'],secilen_test=r['secilen_test'],
                genel_kapsam_yuzde=r['genel_kapsam'],kritik_kapsam_yuzde=100.,on_kosul_ihlali=r['kopuk_on_kosul'],
                arama_sn=r['algoritma_suresi'],tam_paket_sn=np.nan,secili_paket_sn=np.nan,
                planned_evaluations=int(r['aday_degerlendirmesi']) if r['algoritma']!='Greedy' else 1,
                actual_evaluations=np.nan))
        self.result=dict(source='Kayıtlı deney',settings=asdict(Settings.full()),rows=pd.DataFrame(records),
                         runs={},paths=[str(p) for p in paths],note='CSV bütçesi planlanan değerdir. Gerçek çağrı izi, tam/seçili süre toplamı ve seçili test listesi arşivde yok. Eksik alanlar hesaplanmış gibi doldurulmadı.')
        self.result_fingerprint=None
        return self.result

    def run(self, progress=None):
        config=deepcopy(self.settings).validate()
        self.last_error=None
        # Bütün havuzları ilk aramadan önce doğrula.
        pools={(n,s):self.pool(n,s) for n in config.sizes for s in config.data_seeds}
        for pool in pools.values():validate_pool(pool,config.min_coverage)
        fingerprint=self.fingerprint()
        folder=self.root/'sonuclar'/'sunum_calistirmalari'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
        folder.mkdir(parents=True,exist_ok=False)
        manifest=dict(source='Yeni çalıştırma',settings=asdict(config),fingerprint=fingerprint,
                      inputs={f'{n}_{s}':digest_pool(p) for (n,s),p in pools.items()},
                      resolved_options={str(n):config.options(n) for n in config.sizes},
                      budgets={str(n):config.budget(n) for n in config.sizes},
                      python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                      code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'kod/algoritmalar/bmgo.py',ROOT/'kod/algoritmalar/baselines.py',ROOT/'kod/algoritmalar/fitness.py',ROOT/'kod/algoritmalar/izleme.py',ROOT/'kod/veri/data_generator.py']})
        def write_manifest(state,**extra):
            manifest.update(status=state,**extra)
            (folder/'ayarlar.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
        write_manifest('running')
        rows,runs=[],{}
        total=len(pools)*(2*len(config.algorithm_seeds)+1)
        try:
            for (n,ds),pool in pools.items():
                np.savez_compressed(folder/f'girdi_{n}_{ds}.npz',**{k:pool[k] for k in ['coverage','durations','dependencies','critical']})
                (folder/f'veri_{n}_{ds}.json').write_text(json.dumps({k:pool[k] for k in ['names','features','metadata']},ensure_ascii=False),encoding='utf8')
                common=dict(coverage_matrix=pool['coverage'],durations=pool['durations'],dependency_matrix=pool['dependencies'],
                            critical_feature_indices=pool['critical'],min_coverage=config.min_coverage,
                            **{k:getattr(config,k) for k in ['w_time','w_count','w_redundancy','w_rare']})
                budget=config.budget(n)
                for method in ['IBMGO','GA','Greedy']:
                    for seed in ([None] if method=='Greedy' else config.algorithm_seeds):
                        trace={};start=perf_counter()
                        if method=='IBMGO':
                            solution,score,curve=binary_mountain_gazelle_optimizer(**common,**config.options(n),pop_size=config.population,max_iter=config.iterations,random_seed=seed,trace=trace)
                            planned=budget['ibmgo']
                        elif method=='GA':
                            solution,score,curve=genetic_algorithm_optimizer(**common,pop_size=config.population,max_generations=budget['generations'],random_seed=seed,trace=trace)
                            planned=budget['ga']
                        else:
                            solution,score=greedy_regression_selector(**common);curve=[];planned=1
                        elapsed=perf_counter()-start
                        m=get_suite_metrics(solution,pool['coverage'],pool['durations'],pool['dependencies'],pool['critical'])
                        if m['critical_missing'] or m['broken_dependencies'] or m['coverage_ratio']+1e-12<config.min_coverage:
                            raise RuntimeError('Algoritma geçersiz bir paket döndürdü.')
                        order,apfd=prioritize_tests_and_calculate_apfd(solution,pool['coverage'],pool['durations'],dependency_matrix=pool['dependencies'],critical_feature_indices=pool['critical'])
                        key=f'{n}_{ds}_{method}_{seed}'
                        runs[key]=dict(size=n,data_seed=ds,algorithm_seed=seed,method=method,solution=solution.tolist(),order=list(map(int,order)),trace=trace,curve=list(map(float,curve)),arama_sn=elapsed)
                        rows.append(dict(size=n,data_seed=ds,algorithm_seed=seed,method=method,fitness=score,
                            tam_paket_sn=m['total_duration'],secili_paket_sn=m['selected_duration'],tasarruf_yuzde=100*(1-m['time_ratio']),
                            secilen_test=m['selected_count'],kritik_kapsam_yuzde=100*(1-m['critical_missing']/m['critical_total']) if m['critical_total'] else 100.,
                            genel_kapsam_yuzde=100*m['coverage_ratio'],on_kosul_ihlali=m['broken_dependencies'],arama_sn=elapsed,
                            planned_evaluations=planned,actual_evaluations=trace.get('evaluations',1),input_hash=digest_pool(pool)))
                        if progress: progress(len(rows),total,key)
            frame=pd.DataFrame(rows)
            frame.to_csv(folder/'ham_sonuclar.csv',index=False,encoding='utf-8-sig')
            summarize(frame).to_csv(folder/'ozet.csv',index=False,encoding='utf-8-sig')
            (folder/'izler_ve_secimler.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding='utf8')
            write_manifest('complete',completed_runs=len(rows))
        except Exception as exc:
            self.last_error=str(exc);write_manifest('failed',error=str(exc),completed_runs=len(rows))
            raise
        self.result=dict(source='Yeni çalıştırma',settings=asdict(config),rows=frame,runs=runs,
                         pools=pools,paths=[str(folder)],note='Süreler sentetik test sürelerinin toplamıdır. Arama süresi ayrıca ölçülür.')
        self.result_fingerprint=fingerprint
        return self.result


def summarize(rows):
    grouped=rows.groupby(['size','method'],sort=False)
    counts=grouped.agg(kosum_sayisi=('fitness','size'),veri_kumesi=('data_seed','nunique'),algoritma_tohumu=('algorithm_seed','nunique'))
    metrics=grouped[METRICS].agg(['mean','std'])
    metrics.columns=['_'.join(c) for c in metrics.columns]
    return counts.join(metrics).reset_index()


def result_figures(result,size):
    frame=result['rows'];frame=frame[frame['size']==size]
    fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    colors=['#178c86','#5064b9','#c58238']
    for ax,key,label in zip(axes.flat,METRICS,LABELS):
        groups=[frame.loc[frame.method==m,key].dropna() for m in ['IBMGO','GA','Greedy']]
        if not any(len(g) for g in groups):
            ax.text(.5,.5,'Arşivde kayıt yok',ha='center',transform=ax.transAxes)
        else:
            means=[g.mean() for g in groups];errors=[g.std() if len(g)>1 else 0 for g in groups]
            ax.bar(['IBMGO','GA','Greedy'],means,yerr=errors,color=colors,capsize=4)
            ax.set_ylim(bottom=0)
        ax.set_title(label);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(f"{result['source']} · {size} test · Ortalama ± örnek standart sapma\nAynı boyutta veri ve algoritma tekrarları birlikte özetlenir. Greedy her veri için bir kez.",fontsize=12)
    return fig


def convergence_figures(result,key):
    run=result['runs'][key];trace=run['trace']
    if not trace: return None
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,method in zip(axes[:2],['IBMGO','GA']):
        other=next(r for r in result['runs'].values() if r['size']==run['size'] and r['data_seed']==run['data_seed'] and r['algorithm_seed']==run['algorithm_seed'] and r['method']==method)
        t=other['trace']
        # İlk nokta, başlangıç popülasyonu tamamlandıktan sonraki en iyi puan.
        ax.step(range(t['steps']+1),[t['initial_fitness']]+other['curve'],where='post',label=method)
        ax.set_xlabel('İterasyon (0 başlangıç)' if method=='IBMGO' else 'Nesil (0 başlangıç)')
        events=t['events']
        shown=events
        axes[2].step([e['evaluation'] for e in shown],[e['best'] for e in shown],where='post',label=method)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Aynı veri ve boyutta diğer tohumların eğrileri silik çizilir.
        for repeat in result['runs'].values():
            if repeat['size']==run['size'] and repeat['data_seed']==run['data_seed'] and repeat['method']==method and repeat['algorithm_seed']!=run['algorithm_seed']:
                rt=repeat['trace'];ax.plot(range(rt['steps']+1),[rt['initial_fitness']]+repeat['curve'],alpha=.18,color='gray')
    for ax in axes:
        ax.set_ylabel('O ana kadar en iyi fitness ↓');ax.grid(alpha=.2);ax.legend()
    axes[2].set_xlabel('Kümülatif gerçek amaç fonksiyonu çağrısı')
    fig.suptitle(f"Yeni çalıştırma · {run['size']} test · veri {run['data_seed']} · algoritma {run['algorithm_seed']}\nAynı veri ve boyut karşılaştırılır. Yeniden puanlama benzersiz aday sayısı değildir.",fontsize=11)
    return fig


def selected_table(result,key,limit=20):
    r=result['runs'][key];pool=result['pools'][(r['size'],r['data_seed'])]
    records=[]
    for rank,i in enumerate(r['order'][:limit],1):
        parents=np.flatnonzero(pool['dependencies'][:,i])
        records.append(dict(sira=rank,test=pool['metadata']['test_cases'][i]['id'],
                            senaryo=pool['names'][i],sure_sn=pool['durations'][i],
                            on_kosul=', '.join(pool['metadata']['test_cases'][j]['id'] for j in parents)))
    return pd.DataFrame(records)
