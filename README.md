# ML-Ionosphere
Classification of radar returns from the ionosphere using neural networks.

## Wstęp
Sieci neuronowe mają wiele potencjalnych zastosowań w przetwarzaniu sygnałów.

Opisuję tutaj zastosowanie sieci neuronowych do problemu klasyfikacji, który normalnie wymagałby interwencji człowieka. Sieci zostały przeszkolone w zakresie rozróżniania "dobrych" od "złych" danych zwrotnych radaru z jonosfery.

Wykorzystane dane radarowe zostały zebrane przez Space Physics Group of The John Hopkins University Applied Physics Laboratory. System radarowy, znajdujący się w Goose Bay, w Labradorze, składa się z fazowanej macierzy 16 anten o wysokiej częstotliwości, o łącznej mocy transmitowanej rzędu 6,4 kW i zysku anteny około 30 dBm przy zakresach częstotliwości od 8 do 20 MHz. Dane zwracane przez radary są wykorzystywane do badania fizyki jonosfery w warstwach E i F (wysokość od 100 do 500 km).

Radar działa poprzez transmisję wzoru wielowymiarowego do jonosfery. Odbiornik jest włączany pomiędzy impulsami, a docelową prędkość określa się, mierząc przesunięcie fazowe zwrotów. Jeśli oznaczymy odebrany sygnał z impulsu w chwili t przez C (t) = A (t) + iB (t), wtedy funkcja autokorelacji (ACF), R, jest podana przez R (t, k) = {suma i = 0 do 16} C (t + iT) C * [t + (i + k) T], gdzie T jest okresem powtarzania impulsu, k oznacza numer impulsu, a * oznacza sprzężenie złożone.

17 par liczb, reprezentujących 17 dyskretnych wartości części rzeczywistej i odpowiadających im 17 wartości urojonej części ACF, to dane wejściowe do sieci neuronowej.

Sieć, którą wykorzystałem, jest znana jako sieć typu feedforward, które zawierają warstwę wejściową z identycznymi neuronami, warstwą pośrednią lub ukrytą oraz warstwą wyjściową. Wszystkie jednostki są dowolną warstwą połączoną ze wszystkimi jednostkami w powyższej warstwie. Nie ma innych połączeń. Jednostki wejściowe nie wykonują obliczeń, ale służą jedynie do podziału danych wejściowych. Jednostki w ukrytej warstwie nie mają bezpośrednich połączeń ze światem zewnętrznym, ale po przetworzeniu danych wejściowych przekazują swoje wyniki do jednostek warstwy wyjściowej. W procesie zwanym treningiem, sieć otrzymuje wybrane przykłady wejściowe i odpowiednią pożądaną odpowiedź wyjściową (lub cel). Wagi połączeń są zmieniane, przy użyciu algorytmu uczenia się zwanego propagacją wsteczną, aż błąd wyjściowy jest zminimalizowany w sensie najmniejszych kwadratów.

Ten zestaw, który nie ma danych wspólnych z zestawem treningowym, nazywany jest zestawem testowym.

Ponieważ każdy dyskretny zwrot radaru składa się z części rzeczywistej i urojonej, wynik wynosi 34 wartości na ACF. Te 34 wartości służą jako dane wejściowe do sieci. Każde wejście znormalizowano do zakresu [-1, 1]. Liczba ukrytych węzłów była zmieniana od 0 (bez ukrytej warstwy) do 15. Ponieważ sieć jest obecnie używana do klasyfikowania danych wejściowych tylko w dwóch klasach (dobra i zła), potrzebny był tylko jeden węzeł wyjściowy. Ten węzeł wyprowadza 1 dla dobrego powrotu, a 0 dla złego powrotu. Ogólnie dobre wyniki są wskazywane przez dobrze zdefiniowane sygnały, które świadczą o obecności pewnego rodzaju struktury w jonosferze. Złe zwroty mogą być spowodowane
- z powodu braku identyfikowalnej struktury (sygnał przechodzi przez jonosferę),
- przez niespójne rozproszenie (sygnały są odbijane od zbyt wielu struktur, co powoduje anulowanie fazy),
- przez absorpcję impulsów radarowych.

Złe zwroty są bardziej zróżnicowane niż dobre. Ta różnica znajduje odzwierciedlenie w zachowaniu sieci.

## Bibliografia
- Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural  networks. Johns Hopkins APL Technical Digest, 10, 262-266.
