clear all;
Data=load('~/Documents/Data/MultiLabel/WikiLSHTC/wikiLSHTC_train_word.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;


nD = max(Data(:,1)); nV = max(Data(:,2)); 
fprintf(1,'No. of Documents %d, Vocabulary Size %d\n',nD,nV);

val = ones(size(Data,1),1);
X=sparse(Data(:,1),Data(:,2),val,nD,nV);
clearvars Data;
X=logical(X);
X=double(X);

%% Results
%%% Tensor AUC=> 50: 0.843961, 75: 0.860452, 100: 0.867318, 125: 0.867755, 150:0.878713
%%% LEML AUC=> 50: 0.818995, 75: 0.883674, 100: 0.901026 125: 0.937721, 150: 0.948912
%%% y=[0.818995, 0.823674, 0.831026, 0.857721, 0.868912]; x=[0.843961, 0.850452, 0.857318, 0.867755, 0.878713];
%%% Tensor Mem => 50:12G, 75:13G 100:14G, 125:15G, 150:17G
%%% LEML Mem => 50:37G, 75:40.5G 100:45G, 125:47.3G, 150:50G

%% Training on Words%%
t0=tic;
[nD nV]=size(X);
M1 = full(sum(X,1)');
Z1=sum(M1); M1=M1./Z1;

Xsum = full(sum(X,2));
Z2 = sum(Xsum.*Xsum);

%% Whitening
K = 100; %Number of Clusters


[U,S,~]=svds(X'/sqrt(Z2),K);

s=diag(S);


W = U*diag(1./s);
clearvars U;

%% Tensor Factorization
Xsum = full(sum(X,2));
Z3 = sum(Xsum.*Xsum.*Xsum);
Mx=X*W./power(Z3,1/3); W2 = W*inv(W'*W); clearvars W;

G = zeros(K,K,K);
for i=1:K
    for j=1:K
        G(:,i,j)=(Mx(:,i).*Mx(:,j))'*Mx;
    end
    sprintf('Matrix Multiplied: %d',i)
end


% Extract tensor eigenvalues
G=tensor(G);
eigvals = zeros(K,1);
V=zeros(K,K);
for k=1:K
    G=symmetrize(G);
    [s,U]=sshopm(G,'Tol',1e-16);
    if(s>0) eigvals(k)=s; V(:,k)=U;
    else eigvals(k)=-s; V(:,k)=-U;
    end

    G=G-tensor(ktensor(s,U,U,U));
    fprintf(1,'%d th EigenValue Extracted: %f\n',k,s);
end

clearvars G;

V2 = zeros(nV,K); 
for k=1:K
    V2(:,k)=eigvals(k)*W2*V(:,k);
end

clearvars X; 

%% Parameter Extraction for Test Set %%
Data=load('~/Documents/Data/MultiLabel/WikiLSHTC/wikiLSHTC_test_word.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;
val = ones(size(Data,1),1);
nDtest = max(Data(:,1)); 
X=sparse(Data(:,1),Data(:,2),val,nDtest,nV);
clearvars Data;
X=logical(X); X=double(X);


P=normalize_cols(V2);
beta = .001; P = P + beta*ones(size(P))/nV;
for k=1:K
    P(:,k)=P(:,k)./sum(P(:,k));
end

Pu = zeros(K,nDtest);
for u=1:nDtest
    [i j n]=find(X(u,:));
    L = P(j',:);
    Lprob = log(pi) + sum(log(L),1)';
    [~,imax]=max(Lprob);
    ll = Lprob-Lprob(imax);
    prob = exp(ll) + realmin('double');
    prob = prob/sum(prob);
    if ~isempty(find(isnan(prob),1))
        error('NaN in user probability');
    end
    Pu(:,u)=prob;
    if(mod(u,1000)==0) fprintf(1,'Test Document Probability Extracted: %d\n',u); toc(t0);
    end
end
  



filename = sprintf('~/Documents/Data/MultiLabel/WikiLSHTC/WordProb_WikiLSHTC_K%d.txt',K);
dlmwrite(filename,P','delimiter',' ');
filename = sprintf('~/Documents/Data/MultiLabel/WikiLSHTC/DocProb_WikiLSHTC_K%d.txt',K);
dlmwrite(filename,Pu,'delimiter',' ');

clearvars P Pu;


%% Moment on Labels %%
Label=load('~/Documents/Data/MultiLabel/WikiLSHTC/wikiLSHTC_train_label.txt');
Label(:,1)=Label(:,1) + 1; Label(:,2)=Label(:,2)+1;
nL=max(Label(:,2));

fprintf(1,'No. of Documents %d, Label Size %d',nD,nL);

val = ones(size(Label,1),1);
L=sparse(Label(:,1),Label(:,2),val,nD,nL);
clearvars Label;
L=logical(L);
L=double(L);


Ml = Mx*V;
Plabel = L'*(Ml.*Ml);
[i j v]=find(Plabel<0); Plabel(i,j)=0; %Set the negative values to zero

for k=1:K
    Plabel(:,k)=Plabel(:,k)/sum(Plabel(:,k));
end

fprintf(1,'Training Finished\n');
toc(t0);

filename = sprintf('~/Documents/Data/MultiLabel/WikiLSHTC/LabelProb_WikiLSHTC_K%d.txt',K);
dlmwrite(filename,Plabel','delimiter',' ');
clearvars L; clearvars nD;
    
