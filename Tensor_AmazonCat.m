clear all;
Data=load('~/Documents/Data/MultiLabel/AmazonCat/amazonCat_train_word.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;


nD = max(Data(:,1)); nV = max(Data(:,2)); 
fprintf(1,'No. of Documents %d, Vocabulary Size %d\n',nD,nV);

val = ones(size(Data,1),1);
X=sparse(Data(:,1),Data(:,2),val,nD,nV);
clearvars Data;
X=logical(X);
X=double(X);

%% Results
%%% Tensor AUC=>  50: 0.971620, 75:0.973733 ,100: 0.974969, 125:0.976283, 150: 0.977498 (@1K)
%%% Tensor Time => 50: 11G, 75:11G,  100: 12G, 125:12.5G 5625.032672, 150: 5000 s, 13 G
%%% LEML AUC=> 50: 0.977010, 75: 0.976324, 100:0.975836, 125:0.975626, 150:0.973134
%%% LEML Mem=> 50: 22G, 75: 26.6 100: 30G, 125:34G, 150:38G

%% Training on Words%%
t0=tic;
[nD nV]=size(X);
M1 = full(sum(X,1)');
Z1=sum(M1); M1=M1./Z1;

Xsum = full(sum(X,2));
Z2 = sum(Xsum.*Xsum);

%% Whitening
K = 75; %Number of Clusters


[U,S,~]=svds(X'/sqrt(Z2),K);

s=diag(S);
W = U*diag(1./s);

filename = sprintf('~/Documents/Data/MultiLabel/AmazonCat/Eigv_M2_Amazon3M_K%d.txt',K);
dlmwrite(filename,U','delimiter',' ');
filename = sprintf('~/Documents/Data/MultiLabel/AmazonCat/Eigs_M2_Amazon3M_K%d.txt',K);
dlmwrite(filename,s,'delimiter',' ');




%% Tensor Factorization
Xsum = full(sum(X,2));
Z3 = sum(Xsum.*Xsum.*Xsum);
Mx=X*W./power(Z3,1/3);

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

%W2 = W*inv(W'*W);
V2 = zeros(nV,K); W2 = pinv(W');
for k=1:K
    V2(:,k)=eigvals(k)*W2*V(:,k);
end

clearvars X;
toc(t0);

%% Moment on Labels %%
Label=load('~/Documents/Data/MultiLabel/AmazonCat/amazonCat_train_label.txt');
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


filename = sprintf('~/Documents/Data/MultiLabel/AmazonCat/LabelProb_Amazon3M_K%d.txt',K);
dlmwrite(filename,Plabel','delimiter',' ');

clearvars L; clearvars nD; clearvars Plabel;

fprintf(1,'Training Finished\n');
toc(t0);
    
%% Parameter Extraction for Test Set %%
Data=load('~/Documents/Data/MultiLabel/AmazonCat/amazonCat_test_word.txt');
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

Pu = zeros(K,10000); 
for u=1:10000
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
  

filename = sprintf('~/Documents/Data/MultiLabel/AmazonCat/WordProb_AmazonCat_K%d.txt',K);
dlmwrite(filename,P','delimiter',' ');
filename = sprintf('~/Documents/Data/MultiLabel/AmazonCat/DocProb_10K_AmazonCat_K%d.txt',K);
dlmwrite(filename,Pu,'delimiter',' ');

clearvars P Pu;

