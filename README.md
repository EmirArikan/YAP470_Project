# YAP470_Project
## Beyin MRI Görüntülerinden Tümör Derecesi Sınıflandırma

Bu projede, beyin tümörü olan hastalara ait MRI görüntülerinden yola çıkarak, tümörün hangi seviyede (derecede) olduğunu sınıflandırmak amaçlanmaktadır.

### Veri Seti

Projemizde kullanılan veri setinde, beyin tümörü olan hastalara ait MRI görüntüleri ve bu görüntülere karşılık gelen tümör derecelerini içeren bir `.csv` dosyası bulunmaktadır.

-[Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  
-[Brain Tumor Segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

Verimizin ön işlemlerden geçirildikten sonraki son hali şu şekildedir:
![output](https://github.com/user-attachments/assets/8d97c2aa-a44e-4717-9a1a-fedd54c82de7)

### Dosyalar ve Kodlar
- #### modeller
  Önceden eğitilmiş MLP(Multilayer Perceptron), SVM(Support Vector Machine) ve RF(Random Forest) modelleri ve PCA(Principal Component Analysis) modeli bulunmaktadır.

- #### total_data
  Github'a tüm datalar, büyüklüğünden ötürü yüklenemeyeceği için her sınıftan eşit dağılmış 185 resimden oluşan mini bir veri setidir. Bu veri seti ile herhangi bir model eğitilebilir.

- #### test_data
  Modellerin test edilebilmesi için 102 resimden oluşan, sınıfları eşit dağılmış küçük bir veri setidir.

- #### main.ipynb
  Ana 'training' kodu, bu kodda kullanıcı görsellerin olduğu klasörü, `.csv` formatındaki data dosyasını ve modellerin kaydedileceği isimleri belirleyerek baştan bir model eğitip başarı metrikleriyle modeli test edebilirler. 

- #### test.ipynb
  Eğitilen modelleri, yeni görsellerle test edebilen koddur. Kullanıcı görsellerin olduğu klasörü, `.csv` formatındaki data dosyasını ve kullanmak istedikleri modeli girerek, sonuçları gözlemleyebilirler.



### Sonuçlar
#### Model Seçiminin Başarıya Etkisi
Model seçimi, makine öğrenmesi uygulamalarında başarıyı doğrudan etkileyen temel unsurlardan biridir. Ancak model seçimi için tek bir doğru yoktur; bu seçim, kullanılan veri setine ve problemin doğasına bağlı olarak değişkenlik gösterebilir. Bu nedenle farklı modellerin performanslarını karşılaştırmak önemlidir. Yapılan karşılaştırmalar sonucunda, problemimiz özelinde Çok Katmanlı Algılayıcı (MLP) modelinin diğer modellere kıyasla daha başarılı sonuçlar verdiği gözlemlenmiştir.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3e0cb4e1-52f9-493e-a0de-a38e2f0b2145" alt="image" width="700"/>
</p>


#### Önişlemenin Başarıya Katkısı
Ham verinin doğrudan kullanımı, özellikle makine öğrenmesi modellerinde yetersiz sonuçlara yol açabilir. Bu nedenle, veriye önişleme uygulanması performans açısından kritik bir adımdır. Çalışmamızda, modeller hem önişleme uygulanmadan hem de önişleme ile eğitilerek karşılaştırılmıştır. Sonuçlar, önişleme uygulanan verilerle eğitilen modellerin daha başarılı olduğunu göstermiştir. Ancak bazı modellerde, belirli hiperparametre ayarlarında istisnai durumlar da gözlemlenmiştir.

<p align="center">
  <img src="https://github.com/user-attachments/assets/36ed520a-de29-4f96-a7c5-a05869104317" alt="image" width="500"/>
</p>

#### Öznitelik Sayısı ve Seçiminin Etkisi
Model başarısını etkileyen bir diğer önemli unsur, kullanılan özniteliklerin (özelliklerin) sayısı ve kalitesidir. Bu bağlamda, Temel Bileşenler Analizi (PCA) yöntemiyle öznitelik boyutu azaltılmış ve farklı sayılarda öznitelik kullanılarak deneyler gerçekleştirilmiştir. 10, 100 ve 1000 öznitelik ile yapılan denemeler sonucunda; çok az öznitelik kullanıldığında modelin yeterince öğrenemediği, çok fazla öznitelik kullanıldığında ise boyutlanma laneti (curse of dimensionality) nedeniyle performansın düştüğü tespit edilmiştir. En başarılı sonuçlar 100 öznitelik kullanıldığında elde edilmiştir.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f00e61d0-12e0-4c08-9867-f1cf46cd1de6" alt="image" width="700"/>
</p>


#### Hiperparametre Seçimi
Hiperparametrelerin uygun şekilde belirlenmesi, modelin performansını önemli ölçüde etkilemektedir. Bu nedenle, daha verimli sonuçlar elde edebilmek adına farklı hiperparametre kombinasyonları denenmiştir. Hiperparametre optimizasyonu için random search yöntemi kullanılmış ve bu sayede modellerin başarı oranlarında artış sağlanmıştır.
