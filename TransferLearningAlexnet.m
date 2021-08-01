%% Cargamos los datos
% Descomprimimos las imagenes y las guardamos como un almac�n de datos de im�genes 
% llamada |imageDatastore|
% Etiquetamos las imagenes seg�n el nombre de sus carpetas
unzip('MerchData.zip');
images = imageDatastore('MerchData',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%%
% Dividimos los datos de entrenamiento y validaci�n. Utilizamos el 70 % de
% las imagenes para entrenamiento y un 30 % para validaci�n -
% "splitEachLabel" permite dividir las imagenes en dos almacenes nuevos 
[trainingImages,validationImages] = splitEachLabel(images,0.7,'randomized');

%%
% Ahora tenemos 55 im�genes de entrenamiento y 20 de validaci�n, mostramos
% unas imagenes a modo ejemplo
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end

%% Cargamos la red preentrenada Alexnet

net = alexnet;

%% Mostramos la arquitectura de la red. Posee 5 capas convuolcionales y 
%% 3 capas completamente conectadas
net.Layers



%% Transferimos las capas a la nueva red
% Las �ltimas 3 capas de la red preentrenada "net" est�n configuradas para
% 1000 clases, estas 3 capas deben ajustarse para el nuevo problema de
% clasificaci�n, extraemos todas las capas, exceptuando las �ltimas 3 de la
% red preentrenada
layersTransfer = net.Layers(1:end-3);

%%
% Transferimos las capas a la nueva tarea de clasificaci�n reemplazando las
% �ltimas 3 capas por:una capa completamente conectada, una capa softmax y
% una capa de salida de clasificaci�n. Establecemos el mismo tama�o de
% clases para los datos
numClasses = numel(categories(trainingImages.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];



%% Opciones de entrenamiento
% Especificamos las opciones de entrenamiento, para realizar la trasnferen
% de aprendizaje, mantenemos las funciones de las primeras capas de la reed
% preentrenada, establecemos InitialLearnRate a un valor peque�o para
% relantizar el aprendizaje en las capas transferidas, en el paso anterior,
% se aument� la tasa de aprendizaje para la capa completamente conectada
% para acelerar el aprendizaje en las capas finales, esta combinaci�n de
% tasas de aprendizaje da como resultado un aprendizaje r�pido solo en las
% nuevas capas y un aprendizaje m�s lento en las otras.
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);   %% Detiene el ent si la p�rdida de validaci�n deja de mejorar

%% Al realizar un aprendizaje por transferencia, no es necesario entrenar 
% durante tantas �pocas. Una �poca es un ciclo de entrenamiento completo en
% todo el conjunto de datos de entrenamiento, Especificamos el tama�o del
% mini lote y los datos de validaci�n


%%
% Entrenamos la red
netTransfer = trainNetwork(trainingImages,layers,options);


%% Clasificamos las im�genes de validaci�n
predictedLabels = classify(netTransfer,validationImages);

%%
% Mostramos las 4 im�genes de validaci�n con sus etiquetas
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(validationImages,idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end

%%
% Calculamos la precisi�n de la clasificaci�n en el conjunto de validaci�n.
% La precisi�n es la fracci�n de etiquetas que la red predice correctamente
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels)

