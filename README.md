# Projekt z przedmiotu "Sztuczna inteligencja"
Temat: Zrealizować sieć neuronową uczoną algorytmem wstecznej propagacji błędu z przyspieszeniem metodą adaptacyjnego współczynnika uczenia (trainbpa) uczącą się klasyfikowania jonosfery. [Link do zbioru danych wraz z opisem](http://archive.ics.uci.edu/ml/datasets/Ionosphere)

## Spis treści
1. [Opis problemu](#opis_problemu)
2. [Opis sieci neuronowej](#opis_sieci_neuronowej)
3. [Skrypt](#skrypt)
4. [Eksperymenty](#eksperymenty)
5. [Wnioski](#wnioski)

<a name="opis_problemu"></a>
## 1. Opis problemu

Opisuję tutaj zastosowanie sieci neuronowych do problemu klasyfikacji, który normalnie wymagałby interwencji człowieka. Klasyfikacja jest podkategorią **supervised learning**, której celem jest przewidzieć kategoryczne etykiety przypadków bazując na wcześniejszych obserwacjach. Etykiety mają charakter dyskretny i niejako pozwalają podzielić przypadki na grupy. Sieć została nauczona rozróżniania "dobrych" od "złych" danych zwrotnych radaru z jonosfery.

Wykorzystane dane radarowe zostały zebrane przez Space Physics Group of The John Hopkins University Applied Physics Laboratory. System radarowy, znajdujący się w Goose Bay, w Labradorze, składa się z fazowanej macierzy 16 anten o wysokiej częstotliwości, o łącznej mocy transmitowanej rzędu 6,4 kW i zysku anteny około 30 dBm przy zakresach częstotliwości od 8 do 20 MHz. Dane zwracane przez radary są wykorzystywane do badania fizyki jonosfery w warstwach E i F (wysokość od 100 do 500 km).

Radar działa poprzez transmisję wzoru wielowymiarowego do jonosfery. Odbiornik jest włączany pomiędzy impulsami, a docelową prędkość określa się, mierząc przesunięcie fazowe zwrotów. Jeśli oznaczymy odebrany sygnał z impulsu w chwili t przez C (t) = A (t) + iB (t), wtedy funkcja autokorelacji (ACF), R, jest podana przez R (t, k) = {suma i = 0 do 16} C (t + iT) C * [t + (i + k) T], gdzie T jest okresem powtarzania impulsu, k oznacza numer impulsu, a * oznacza sprzężenie złożone.

17 par liczb, reprezentujących 17 dyskretnych wartości części rzeczywistej i odpowiadających im 17 wartości urojonej części ACF, to dane wejściowe do sieci neuronowej.

W procesie zwanym treningiem, sieć otrzymuje wybrane przykłady wejściowe i odpowiednią pożądaną odpowiedź wyjściową (lub cel). Wagi połączeń są zmieniane, przy użyciu algorytmu uczenia się zwanego propagacją wsteczną, aż błąd wyjściowy jest zminimalizowany w sensie najmniejszych kwadratów.

Ponieważ każdy dyskretny zwrot radaru składa się z części rzeczywistej i urojonej, wynik wynosi 34 wartości na ACF. Te 34 wartości służą jako dane wejściowe do sieci. Każde wejście znormalizowano do zakresu [-1, 1]. Ogólnie dobre wyniki są wskazywane przez dobrze zdefiniowane sygnały, które świadczą o obecności pewnego rodzaju struktury w jonosferze. Złe zwroty mogą być spowodowane:
- brakiem identyfikowalnej struktury (sygnał przechodzi przez jonosferę),
- niespójnością rozproszenia (sygnały są odbijane od zbyt wielu struktur, co powoduje anulowanie fazy),
- absorpcją impulsów radarowych.

Zbiór składa się z 351 przypadków. W zbiorze znaleziono jeden powtarzający się przypadek, co skutkowało jego eliminacją. Ponadto druga kolumna zawiera tylko jedną wartość będącą zerem.

Dane zostały podzielona na dwa zbiory:
- zbiór danych uczących jako 200 przypadków po 100 dla każdej kategorii,
- zbiór do testowania sieci liczący 150 przypadków (125 "dobrych" i 25 "złych").

<a name="opis_sieci_neuronowej"></a>
## 2. Opis sieci neuronowej

**Sieć neuronowa** to system przeznaczony do przetwarzania informacji, którego budowa i zasada działania są w pewnym stopniu wzorowane na funkcjonowaniu fragmentów rzeczywistego (biologicznego) systemu nerwowego. Wyróżniająca cechą sieci neuronowej jako narzędzia informatycznego jest możliwość komputerowego rozwiązywania przy jej pomocy praktycznych problemów bez ich uprzedniej matematycznej formalizacji. Najbardziej znamienną cechą sieci neuronowej jest jej zdolność uczenia się na podstawie przykładów i możliwość automatycznego uogólniania zdobytej wiedzy. **Uczenie** jest to proces oparty na prezentacji **przypadków uczących** (przykładów prawidłowo rozwiązanych zadań) należących do **zbioru uczącego**. W trakcie tych pokazów następuje stopniowe dopasowywanie się sieci do tego, by nabyła ona umiejętność rozwiązywania tych zadań. Dopasowywanie to opiera się na porównywaniu odpowiedzi udzielanych przez sieć z odpowiedziami wzorcowymi. Wprowadzana korekta błędu powoduje, że sieć po każdej prezentacji zwiększa szansę udzielenia odpowiedzi bardziej zbliżonej do odpowiedzi wzorcowej. Sens użycia sieci neuronowej polega na tym, ze musi ona (po nauczeniu) rozwiązywać zadania podobne do tych, na których była uczona – ale nie identyczne z nimi. Takie przeniesienie nabytej wiedzy na nowe przypadki nazywane jest **generalizacją**. Zagrożeniem dla generalizacji jest **przeuczenie**.

Gdy sieć jest **przeuczona** – następuje nadmierne dopasowanie jej zachowania do nieistotnych szczegółów konkretnych przypadków uczących – nie mających istotnego znaczenia z punktu widzenia istotnych cech rozwiązywanego zadania.

Sieci neuronowe budowane są zazwyczaj w taki sposób, ze przepływ sygnałów odbywa się w nich wyłącznie w kierunku od wejścia (poprzez ewentualne warstwy ukryte) do wyjścia. Sieci spełniające wyżej podany warunek nazywane są **sieciami jednokierunkowymi** albo sieciami typu **feedforward**.

W trakcie **procesu uczenia sieci** trzeba obserwować postęp tego procesu, ponieważ istnieje ryzyko, że nie przyniesie on pożądanego efektu. Najwygodniej jest to zrobić obserwując, jak zmienia się wartość błędu w kolejnych **epokach procesu uczenia**. Ponieważ sieć może mieć wiele wyjść (powiedzmy, że jest ich M), a epoka składa się z R przypadków uczących – trzeba brać pod uwagę błąd całościowy, sumowany po wszystkich przypadkach uczących i po wszystkich wyjściach sieci. Zwykle przed zsumowaniem wartości błędów podnoszone są do kwadratu, żeby uniknąć efektu kompensowania błędów ujemnych przez błędy dodatnie, a ponadto operacja podnoszenia do kwadratu powoduje silniejsze zaakcentowanie dużych błędów przy równoczesnym zmniejszeniu wpływu błędów małych. Powstający wskaźnik nazywany jest **SSE (Sum Square Errors)** i jest wyrażany wzorem:

![SSE](https://github.com/gumienny/ML-Ionosphere/blob/master/img/SSE.png)

gdzie d_pk oznacza wzorcową odpowiedź, jaka powinna pojawić się przy prezentacji przypadku uczącego o numerze p na wyjściu sieci o numerze k, a y_pk oznacza wartość, jaka się w rzeczywistości pojawiła na tym wyjściu.

Podczas uczenia sieci neuronowej trzeba wykonać bardzo wiele kroków algorytmu uczenia zanim błąd stanie się akceptowalnie mały. Tymczasem zbiór uczący zawiera zwykle ograniczoną liczbę przypadków uczących. Zatem zbiór uczący musi być wykorzystywany w procesie uczenia wielokrotnie. Dla zaznaczenia tego faktu wprowadzono pojęcie **epoki**, rozumiejąc pod tym pojęciem jednorazowe użycie w procesie uczenia wszystkich przypadków uczących zawartych w zbiorze uczącym. Po wykonaniu wszystkich kroków należących do jednej epoki algorytm uczący dokonuje oceny zdolności sieci do generalizacji wyników uczenia przy wykorzystaniu zbioru walidacyjnego. Po stwierdzeniu, że zarówno błąd obliczany na zbiorze uczącym, jak i błąd wyznaczony dla zbioru walidacyjnego nadal jeszcze obiecująco maleją – algorytm uczący wykonuje następna epokę. W przeciwnym przypadku proces uczenia zostaje zatrzymany.

Podstawowym elementem budującym strukturę sieci neuronowej jest **neuron**. Jest to element przetwarzający informacje, w pewnym stopniu wzorowany na funkcjonowaniu biologicznej komórki nerwowej, ale bardzo uproszczony. W strukturze neuronu odnaleźć można wiele wejść oraz jedno wyjście. Ważnym składnikiem neuronu jest komplet wag, których wartości decydujące o zachowaniu neuronu zazwyczaj ustalane są w trakcie procesu uczenia. Zwykle wagi dopasowuje w całej sieci używany algorytm uczenia. Komplet wartości wag ustalonych we wszystkich neuronach w trakcie uczenia determinuje wiedzę, jaka posiada sieć neuronowa. Wagi muszą mieć nadane wartości początkowe, żeby można je było w procesie uczenia poprawiać. To nadawanie wartości początkowych nazywa się **inicjalizacją wag** i polega na tym, że wagom nadaje się wartości losowe.

![model neuronu](https://github.com/gumienny/ML-Ionosphere/blob/master/img/model_neuronu.png)

W neuronie wykonywane są zwykle dwie czynności: **agregacja danych wejściowych** (z uwzględnieniem wag) oraz generacja sygnału wyjściowego (danej wyjściowej). Ze względu na sposób agregacji oraz formę **funkcji aktywacji** wyróżnia się różne typy neuronów. Najczęściej stosowane są neurony liniowe, neurony sigmoidalne i neurony radialne.

Agregacja danych wejściowych to pierwsza czynność, jaką wykonuje neuron. Ponieważ neuron ma zwykle wiele wejść i jedno wyjście – konieczne jest przekształcenie wielu danych wejściowych w jeden wypadkowy sygnał sumarycznego pobudzenia, który kształtuje potem sygnał wyjściowy neuronu za pośrednictwem wybranej **funkcji aktywacji**.

Po agregacji danych wejściowych z uwzględnieniem wag powstaje sygnał sumarycznego pobudzenia. Rola **funkcji aktywacji** polega na tym, że musi ona okreslić sposób obliczania wartości sygnału wyjściowego neuronu na podstawie wartości tego sumarycznego pobudzenia.

**Wsteczna propagacja błędów** jest zasadą ustalania wartości błędów dla neuronów należących do ukrytych warstw wykorzystywana przez algorytm uczenia sieci neuronowej. Wsteczna propagacja błędów jest wykorzystywana przez algorytm **backpropagation**. Algorytm ten opiera się na koncepcji poprawiania na każdym kroku procesu uczenia wartości korekty wag na podstawie oceny błędu popełnianego przez każdy neuron podczas uczenia sieci. Konieczność stosowania wstecznej propagacji błędu wynika z tego, ze tylko błędy w neuronach warstwy wyjściowej wyznacza się bezpośrednio na podstawie danych wyjściowych i odpowiedzi wzorcowych zawartych w zbiorze uczącym. Natomiast dla neuronów w warstwach ukrytych błąd musi być wyznaczany właśnie poprzez wsteczną propagację. Przy tej wstecznej propagacji rozważany neuron otrzymuje wartość błędu wyliczaną na podstawie wartości błędów wszystkich tych neuronów, do których wysyłał on wartość swojego sygnału wyjściowego jako składnika ich danych wejściowych. Przy obliczaniu wartości wstecznie rzutowanego błędu uwzględnia się wartości wag połączeń pomiędzy rozważanym neuronem i neuronami, których błędy są do niego wstecznie rzutowane (wstecznie, bo przeciwnie do kierunku przepływu sygnału w jednokierunkowej sieci). **Quickpropagation** jest odmianą algorytmu wstecznej propagacji błędu, która dzięki dostosowywaniu w poszczególnych krokach procesu uczenia wielkości współczynnika uczenia do lokalnych właściwości funkcji błędu pozwala na znaczne przyspieszenie procesu uczenia.

**Współczynnik uczenia** (learning rate) jest parametrem wiążącym lokalne właściwości funkcji błędu sieci neuronowej, wyznaczane na przykład z pomocą algorytmu wstecznej propagacji błędu, odwołujące się do nieskończenie małych zmian wag z działaniem polegającym na bardzo małych zmianach wag w każdym kolejnym kroku uczenia. Algorytm uczenia wskazuje, w jakim kierunku należy zmienić wagi, żeby błąd popełniany przez sieć zmalał, natomiast wybór współczynnika uczenia decyduje o tym, jak bardzo zdecydujemy się te wagi we wskazanym kierunku zmienić. Jeśli współczynnik uczenia wybierzemy zbyt mały, to proces uczenia może bardzo długo trwać, bo będziemy bardzo wolno zmierzać do finalnego (optymalnego) zestawu wartości wszystkich wag. Jeśli jednak zastosujemy zbyt duży współczynnik uczenia – to będziemy wykonywać zbyt duże kroki i może się zdarzyć, że „przeskoczymy” właściwą drogę zmierzającą do punktu zapewniającego minimum funkcji błędu. W efekcie błąd po wykonaniu poprawki wag może być większy, a nie mniejszy niż poprzednio.

<a name="skrypt"></a>
## 3. Skrypt

<a name="eksperymenty"></a>
## 4. Eksperymenty


#### Już dla trzech neuronów w warstwie ukrytej sieć pozwala na poprawną klasyfikację ponad 97% prezentacji. W tabeli przedstawiono zestaw wag.

|nr wagi|1|2|3|
|---|---|---|---|
|bias|0.6333834181505871|0.5326418414506059|-0.6660517721638104|
|1|-1.0854829259976733|-0.7781708733319381|1.8720940144840768|
|2|0.13225944681833868|-0.1793907505965444|-0.45429977009112865|
|3|-1.149724871947237|-0.20641630399172187|1.6565549080323905|
|4|-1.4807786484377499|0.6827331866752079|0.7556749356441259|
|5|-3.0309682398286855|-2.025076825550214|0.47426891780745056|
|6|1.1408338236373667|-1.3543042230085762|2.1148043291635212|
|7|-0.91041754751011|-1.4386310618280775|1.1512806116280967|
|8|-1.407537671380742|-0.8246232128135239|0.5464419084250073|
|9|-0.39190733037184344|-0.13730255165293123|1.2243943739711585|
|10|-1.0422117081364712|0.7411917601933952|-0.6636290588044952|
|11|0.526040811849955|0.4393189816440608|-2.188920045467482|
|12|-2.142786261342884|0.8728847342228245|-1.6190079503829244|
|13|-0.6276051405375357|-0.7539964606766065|0.7854054732575022|
|14|-1.0304547241847657|0.5044481856891722|-1.0125556641766031|
|15|-0.43352419545259807|-0.4396389183595604|-0.26571775323698865|
|16|-1.4078913477140629|-0.688825879246325|-0.9298374393778075|
|17|-0.7313796477594694|0.4801956303710219|-0.5914183876483475|
|18|-0.12335193701639426|-0.6723087285550347|1.6845637060917955|
|19|0.7086719333249305|0.35544860804678025|0.40445937702025053|
|20|-0.3741273345790994|-0.6511658745347569|-0.08494621047927854|
|21|-0.7101629422326069|-0.5017647559206028|0.855276423889629|
|22|1.257564125401019|1.0127933849437913|-2.075415731199068|
|23|-1.6067337840568727|-0.30826182942443625|0.11457178080088797|
|24|-0.6994653068868214|2.099112891761221|0.49459692400770744|
|25|0.12652540408616883|-0.5283586574628547|0.1863759181662877|
|26|-0.2515631072928051|-0.27328015011147055|1.3188851314336851|
|27|1.4487464639321928|1.0924606294688444|-3.784308937854517|
|28|-2.110630094850287|0.5599615073046283|-1.400980526830546|
|29|0.42561543078594816|-0.03851504640833537|1.416567226621939|
|30|-0.5490006879479632|0.33653539047532915|1.9403351247312348|
|31|-0.015790849742116777|-0.564332855021361|1.0927650140331637|
|32|1.1013534247836412|-0.8875983470357233|-0.5979474300480196|
|33|1.0342760586404058|0.19406292405813394|-0.5245345067308886|
|34|0.23358318998229932|-0.5352795164074374|-1.0599544067111708|

#### Wagi dla warstwy wyjściowej

|nr wagi|1|
|---|---|
|bias|-2.265560894325369|
|1|-6.511229950554139|
|2|-3.9549884790697982|
|3|5.764746888569693|

<a name="wnioski"></a>
## 5. Wnioski

## Bibliografia
- Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989). **Classification of radar returns from the ionosphere using neural  networks**. Johns Hopkins APL Technical Digest, 10, 262-266.
- J. Żurada, M. Barski, W. Jędruch. **Sztuczne sieci neuronowe**. **Podstawy teorii i zastosowania**. Wydawnictwo Naukowe PWN, 1996, 83-01-12106-8
- R. Tadeusiewicz, M. Szaleniec. **Leksykon sieci neuronowych**. Wydanie I, Wrocław 2015, 978-83-63270-10-0
- Sebastian Raschka. **Python Machine Learning**. **Unlock deeper insights into machine learning with this vital guide to cutting-edge predictive analytics**. 978-1-78355-513-0
