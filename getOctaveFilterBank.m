 function [H,fcf]=getOctaveFilterBank(fs, Nfft, Nfpo, fmin_user, fmax_user, edges)
%function [H,fcf]=getOctaveFilterBank(fs, Nfft, Nfpo, fmin_user, fmax_user, edges)
%
%getOctaveFilterBank() gera um banco de filtros triangulares em escala logarítmica com "Nfpo" filtros
% por oitava. Retorna a s respostas em frequência dos fitros na matriz "H" bem como os valores das
% frequências centrais (vértice dos triângulos dos filtros).
% 
%Argumentos:
%   fs   - frequência de amostragem do sinal de áudio. (default: 16000)
%   Nfft - Número de amostras da FFT (para amostrar a resposta em frequência dos filtros)(default=4096)
%   Nfpo - Número de filtros por oitava (default: 12)
%   fmin_user - frequência mínima que o utilizador propõe (default: 32.7032 Hz: C1)
%   fmax_user - frequência máxima que o utilizador propõe (default: 7040 Hz:    A8)
%   edges - string; se igual a 'edges', junta ao vetor de saída fcf, os vértices do 1º e último filtro.
%           (default: eges='')
%Saídas:
%   H    - matriz de dimensão Nb x Nfft/2+1 com a parte positiva das respostas em frequência dos Nb 
%          filtros (um filtro por linha de H)
%   fcf  - vetor com os vétices dos filtros. Se edges='edges' junta fmin e fmax: fcf=[fmin,fcf,fmax]
%
%Nota: os filtros estão sobrepostos em frequência de forma que se fcf(i) for a frequência central do
%filtro i, é também o vértice superior do filtro i-1 e o vértice inferior do filtro i+1.
%A frequência fcf(i) é dada por: 
%                               f(i) = f(0)*2^(i/Nfpo), 
% de forma que f(0)=fmin, f(1) é a frequência central do 1º filtro, etc. f(Nfpo) = 2*f(0) e f(Nb+1)=fmax.
%
%Nota: as frequências centrais são ajustadas, sempre que possível, aos semitons musicais na escala
%temperada. Daí que fmin_user é apenas indicativa e pode ser diferente de fmin. O mesmo para fmax.
%
%Inicialmente é feito o cálculo do número de filtros (Nb) que o banco de
%filtros terá pelas frequências miníma (fmin_user) e máxima (fmax_user)
%dadas pelo utilizador e pelo número de filtros por oitava (Nfpo).
% 
%   O primeiro filtro tem a frequência mínima (fmin), a primeira frequência central (f1).
% 
%   A construção do Banco de Filtros dependerá de fi. Pelo que o
%primeiro valor de fi, será a frequência mínima do primeiro filtro que será
%usado na aplicação do Banco de Filtros, a frequência seguinte no vetor
%fi será a frequência central (fcf) do primeiro filtro. A terceira
%frequência do vetor fi será a frequência alta do primeiro filtro. O
%segundo filtro terá como frequência mínima a frequência central do
%primeiro filtro e terá como frequência central a frequência mais alta
%do primeiro filtro.

% Respostas dos filtros
% 1          .
%           /|\
%          / |  \                    H(1,:)
% --------+  |   +--------------------------------->f
% 1    fmin  f1  .
%               /|\
%             /  |  \                H(2,:)
% -----------+   |   +----------------------------->f
%                f2
% ...
%                  1            .
%                             ´ | `
%                         ´     |    `      H(Nb,:)
%----------------------+        |        +--------->f
%                   f_Nb-1    f_Nb      fmax


	%default values:
	if nargin < 1
		fs=16000;
	end
	if nargin < 2
		Nfft=4096;
	end
	if nargin < 3
		Nfpo=12;
	end	
	if nargin < 4
		fmin_user= 440*2.^(((1+1)*12+0-69)/12);
	end	
	if nargin < 5
		fmax_user= 440*2.^(((8+1)*12+9-69)/12);
	end	
	if nargin < 6
		edges='';
	end	

	if fmax_user>fs/2
		fmax_user=fs/2;
	end


    %if(fmax_user<fs/2)
    %    disp('A frequência máxima tem de ser maior ou igual que metade da frequência de amostragem.');
    %    return;
    %end
    %Numero de linhas de H=Nb
    %size(H)=Nb*(Nfft/2+1)

    %fs=16e3; Nfft=4096; Nfpo=12; fmin_user=50; fmax_user=8e3;
    %fs=16e3; Nfft=4096; Nfpo=7; fmin_user=30; fmax_user=4.1860e3;
    %fs=16e3; Nfft=4096; Nfpo=2; fmin_user=50; fmax_user=8e3;

    i0=ceil(Nfpo*log2(fmin_user/440));
    %i0_indexes=i0+69; %69, Lá como referência

    fmin=440*2^(i0/Nfpo); %Freq. Low do 1ºfiltro - não conta
    
    i=floor(-i0+Nfpo*(log2(fmax_user/440)));
    Nb=i-1;
    fmax=440*2^((i+i0)/Nfpo);

    %f1=440*2^((i0+1)/Nfpo)
    

	i=0:Nb+1;
	fi = fmin*2.^(i/Nfpo);

    % fi=zeros(1,Nb+2);
    % %fi(1)=fmin;
    % for i=0:Nb+1
    %     fi(i+1)=440*2^((i0+i)/Nfpo);
    %     i=i+1;
    % end
    % %f(end)=fmax;

    
    %Construção do Banco de Filtros
    k=0:Nfft/2; f=k*fs/Nfft;
    H=zeros(Nb,Nfft/2+1);
    fcf=zeros(1, Nb);
    for j=1:Nb
        fLow=fi(j); fcf(j)=fi(j+1); fUpp=fi(j+2);

        H(j,:)= (f-fLow)/(fcf(j)-fLow).*(f>fLow & f<=fcf(j)) + ...
			    (f-fUpp)/(fcf(j)-fUpp).*(f>fcf(j)  & f<=fUpp);
    end


    %figure ;
    %plot(f,H)

    %figure ;
    %plot(f, sum(H))

    if nargout>1 && strcmp(edges,'edges')
        fcf=[fmin,fcf,fmax]; %edges
    end

end
