% Limpeza 
clc, clear, close all;

% Iniciar o cronômetro para medir o tempo total de execução
total_start_time = tic;

% Leitura do arquivo
data = readtable('diabetespredictiodataset_pronto.csv');

% Extrair cada coluna de data
index               = data(:,1);
gender              = data(:,2);
age                 = data(:,3);
hypertension        = data(:,4);
heart_disease       = data(:,5);
smoking_history     = data(:,6);
bmi                 = data(:,7);
HbA1c_level         = data(:,8);
blood_glucose_level = data(:,9);
diabetes            = data(:,10);

% Inicializar a matriz de pesos com zeros
pesos = zeros(size(smoking_history, 1), 4); % Alterado para 4 colunas

% Atribuir pesos para 'smoking_history'
for i = 1:size(smoking_history, 1)
    switch char(smoking_history{i, 1})
        case 'ever'
            pesos(i,1) = 5;
        case 'current'
            pesos(i,1) = 4;
        case 'not current'
            pesos(i,1) = 3;
        case 'former'
            pesos(i,1) = 2;
        case 'never'
            pesos(i,1) = 1;
        case 'No Info'
            pesos(i,1) = 0;
    end
end

% One-hot encoding para 'gender'
for i = 1:size(gender, 1)
    switch char(gender{i,1})
        case 'Male'
            pesos(i,2) = 1;
        case 'Female'
            pesos(i,3) = 1;
        case 'Other'
            pesos(i,4) = 1;
    end
end

% Converter table para array
index               = table2array(index);
age                 = table2array(age);
hypertension        = table2array(hypertension);
heart_disease       = table2array(heart_disease);
bmi                 = table2array(bmi);
HbA1c_level         = table2array(HbA1c_level);
blood_glucose_level	= table2array(blood_glucose_level);
diabetes            = table2array(diabetes);
smoking_history     = pesos(:,1);
gender              = pesos(:,2);

% Construção da Matriz P e T
P = [gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level];
T = diabetes;

% Normalizar os dados para o intervalo [0, 1]
P_normalized = normalize(P, 'range');

% Dividindo os dados em conjuntos de treino, validação e teste
[trainInd,valInd,testInd] = dividerand(size(P_normalized,1),0.7,0.15,0.15);

% Separando os conjuntos de treino, validação e teste
P_train = P_normalized(trainInd,:)';
T_train = T(trainInd)';
P_val = P_normalized(valInd,:)';
T_val = T(valInd)';
P_test = P_normalized(testInd,:)';
T_test = T(testInd)';

% Transposta das matrizes
P = P_normalized';
T = T';

% Definir a arquitetura da rede neural
% hidden_units = 32;
hidden_unit = 32;
output_unit = 1;

% Criar a Rede Neural
% net = feedforwardnet([hidden_units, hidden_units, hidden_units]);
net = feedforwardnet([hidden_unit, output_unit]);

% Definir a função de treino
net.trainFcn = 'trainlm';

% Configurar função de ativação para cada camada
net.layers{1}.transferFcn = 'poslin'; 
net.layers{2}.transferFcn = 'logsig'; 
% net.layers{3}.transferFcn = 'poslin'; 

% Definir opções de treinamento
net.trainParam.epochs = 1000; % Número máximo de épocas de treinamento
net.divideFcn = 'divideind'; % Usar divideind para dividir os dados

% Definindo os índices de divisão
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% Iniciar o cronômetro para medir o tempo de treinamento da rede neural
training_start_time = tic;

% Treinar a rede neural com o conjunto de treino
[net, tr] = train(net, P_train, T_train);

% Parar o cronômetro para medir o tempo de treinamento da rede neural
training_end_time = toc(training_start_time);

% Fazer previsões usando a rede neural
saida = net(P_test);

% Plot dos resultados
figure;
plot(1:length(T_test'), T_test', 'bo-', 1:length(saida), saida, 'rx-');
xlabel('Amostras');
ylabel('Diabetes');
title('Previsão de Diabetes: Real vs. Previsto');
legend('Real', 'Previsto');
grid on;

% Cálculos
N = length(T_test);
saida2 = hardlim(saida-0.5);
erro = mse(saida2, T_test);                  % Erro Quadrático Médio (MSE)
errados = sum(abs(T_test - saida2));         % Previsões incorretas
accuracy = (N - errados) / N * 100;     % Precisão do modelo

% Mostrar em tela os valores calculados
disp(['Erro Quadrático Médio (MSE): ', num2str(erro)]);
disp(['Número de previsões incorretas: ', num2str(errados)]);
disp(['Precisão do modelo: ', num2str(accuracy), '%']);

% Plot da matriz de confusão
figure;
plotconfusion(T_test, saida2);

% Matriz de confusão
[stats] = statsOfMeasure(confusionmat(T_test,saida2),1);

% Parar o cronômetro para medir o tempo total de execução
total_end_time = toc(total_start_time);

% Exibir os tempos de execução
disp(['Tempo total de execução: ', num2str(total_end_time), ' segundos']);
disp(['Tempo de treinamento da rede neural: ', num2str(training_end_time), ' segundos']);