%% script de uso da função
%
% getOctaveFilterBank
%
% Gera um sonograma DFT e CQT em escala logarítmica em oitavas.

clear
%Sinal de áudio:
[x,fs_orig]=audioread('Diatonic_scale_on_C.wav');

fs_orig %44100Hz

plot(x(:,1));
%ylabel('Magnitude');xlabel('time (s)');title('Audio Signal');

%passar para 16 kHz o canal esquerdo
fs=16000;
x16 = resample(x(:,1),fs,fs_orig);
soundsc(x16,fs)


Nx=length(x16) %122672 amostras
Tx=Nx/fs % 7.7 segundos

N = 4096; %N/fs=256 ms
M = 800;  % 50ms de hop size
%passar a tramas
xframes=buffer(x16,N,N-M,'nodelay');
Nframes=size(xframes,2); %Ncols = 150
%deve dar o mesmo que
ceil((Nx-N)/M)+1 %pois "buffer" considera sempre uma última trama juntando zeros


%% para sonograma DFT:
Nfft=2^nextpow2(N) %Nfft=N
Xframes=fft(xframes,Nfft);
%fazer vetor de bins da FFT em Hertz (só parte positiva do espetro)
k=0:Nfft/2; %Nfft/2+1 edges. Existem Nfft/2 bins a mostrar. 
% Exemplo Nfft=4; f=[0,D,fs/2]; D=fs/4 e 2 bins: de 0-D e de D-fs/2.
Delta = fs/Nfft; %binwidth da DFT.
%f=k*fs/Nfft; %w=k*2*pi/Nfft
f=(k-1/2)*Delta; %decrescer f de Delta/2, para que o 1º pixel indique entre f=-Delta/2 e f=Delta/2 
f(1)=f(2)/4; %tornar o 1º valor de f positivo por causa de log2(f).


Xframes=abs(Xframes(1:Nfft/2+1,:));

log2f = log2(f); %log2f(1)=-inf

%colocar apenas 80 dB de gama dinâmica:
th=floor(max(max(db(Xframes))) - 80);

t=1:Nframes; %tempo = índice das tramas. Multiplicar por M para ter amostras ou por M/fs para seg.
% eventualmente somar N/2/fs para que o tempo das frames corresponda ao meio das janelas.
figure(1)
surf(t,log2f,max(db(Xframes),th),'EdgeColor','none')
view([0,90]); colorbar
title('DFT sonogram','FontSize', 20)
fnotes=440*2.^(([60    62     64    65    67    69    71 72]-69)/12)
hold on
for n=1:8
	plot3([1,Nframes],log2([fnotes(n),fnotes(n)]),[40,40],'w'); %põe as linhas no topo das montanhas e não na base
end
hold off
ylabel('log_2(f)','FontSize', 16); xlabel('frame index')
%Nota: se se usar plot (em vez de plot3) equivale a tomar z=0 (em vez de 40dB)
%axis([1, 150, 7.7, 9.3]);

%com escala logarítmica em Hz:
figure(1)
surf(t,f,max(db(Xframes),th),'EdgeColor','none')
view([0,90]); colorbar
set(gca,'YScale','log') %põe y com escala logarítmica
title('DFT Sonogram','FontSize', 16)
%Colocar no gráfico as 8 notas centrais da escala temperada (oitava 4)
fnotes=440*2.^(([60    62     64    65    67    69    71 72]-69)/12)
hold on
for n=1:8
	plot3([1,Nframes],[fnotes(n),fnotes(n)],[40,40],'w'); %as notas em Hz (a escala é que é log10)
end
hold off
ylabel('f [Hz]','FontSize', 20); xlabel('frame index')
axis([1, 150, 200, 625]);

%Notar que a resolução espetral aumenta com a frequência: de 4000Hz a 8000Hz temos tantos pontos
%quantos de zero a 4000

%fazendo zoom vemos que o piano estava bem afinado.
%Notar que o 2º harmónico da 1ª nota está síncrona com o 1º harmónico da última nota: um dó uma oitava acima.
figure(2)
plot(k,db(Xframes(:,1))) %1ª trama (já tem o Dó4). O Dó4 aparece no bin 67
plot(f,db(Xframes(:,1))) %a que corresponde a frequência 261.7188 Hz: 67*fs/Nfft. 
% Portanto a nota está entre os bins 66 e 67, entre f=257.8125 e f=261.7188 Hertz. De notar que
% f(Dó4)=fnotes(1)=261.6256 Hz, quase no topo do bin.
% O problema é que imagesc() o pixel 1 fica com o valor de X(1). Para o mostrar a meio do bin,
% devemos decrescer f de Delta/2.

%Afinal o piano estava afinado. Antes de decrescer f de Delta/2, aparecia no bin seguinte

%% CQT

[H,fcf]=getOctaveFilterBank(fs, Nfft, 12, 32, 4500,'edges');
%plot dos filtros
%fcf traz [fmim,fcf,fmax]
Nb = length(fcf)-2
figure(2);
for k=1:Nb
	plot([0,fcf(k),fcf(k+1),fcf(k+2),fs/2],[0,0,1,0,0]); hold on
end
hold off
xlabel('f [Hz]'); axis([0,fs/2,-0.2,1.2])

%Os filtros definidos em H (com amostragem nos bins da DFT)
figure(3)
plot(f,H)
xlabel('f [Hz]'); axis([0,fs/2,-0.2,1.2])
title(sprintf('N_b= %d filtros',Nb))

%As Nb=84 "energias" dos filtros (realmente são valores RMS)
RMS = H*Xframes; % (84 x 2049)x(2049 x 150)

figure(4)
th2=floor(max(max(db(RMS))) - 20);
%neste caso não é preciso surf pois a escala de frequências (dos 84 valores) já está em log2(f).
%imagesc(t,log2(fcf(2:end-1)),max(db(RMS),th2)); axis xy; colorbar
imagesc(t,fcf(2:end-1),max(db(RMS),th2)); axis xy; colorbar
set(gca,'YScale','log')

xlabel('frame index'), ylabel('f [Hz]', 'FontSize', 20);
title('CQT Sonogram with 12 Filters/Oct', 'FontSize', 16)
%title(sprintf('sonograma CQT com %d valores',Nb))
%Colocar no gráfico as 8 notas centrais da escala temperada (oitava 4)
fnotes=440*2.^(([60    62     64    65    67    69    71 72]-69)/12)
hold on
for n=1:8
	%plot([1,Nframes],log2([fnotes(n),fnotes(n)]),'w');
	plot([1,Nframes],[fnotes(n),fnotes(n)],'w');
end
hold off
axis([1, 150, 200, 650]);

