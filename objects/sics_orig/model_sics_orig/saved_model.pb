��
��
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
7
Square
x"T
y"T"
Ttype:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ÿ
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:P@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
�
clustering/clustersVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameclustering/clusters
{
'clustering/clusters/Read/ReadVariableOpReadVariableOpclustering/clusters*
_output_shapes

:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P@**
shared_nameSGD/dense/kernel/momentum
�
-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum*
_output_shapes

:P@*
dtype0
�
SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameSGD/dense/bias/momentum

+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_nameSGD/dense_1/kernel/momentum
�
/SGD/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/kernel/momentum*
_output_shapes

:@*
dtype0
�
SGD/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_1/bias/momentum
�
-SGD/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/bias/momentum*
_output_shapes
:*
dtype0
�
 SGD/clustering/clusters/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" SGD/clustering/clusters/momentum
�
4SGD/clustering/clusters/momentum/Read/ReadVariableOpReadVariableOp SGD/clustering/clusters/momentum*
_output_shapes

:*
dtype0

NoOpNoOp
�"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�"
value�"B�" B�"
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
clusters
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
�
%iter
	&decay
'learning_rate
(momentummomentumEmomentumFmomentumGmomentumHmomentumI*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
�
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

.serving_default* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__
4activity_regularizer_fn
*&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEclustering/clusters8layer_with_weights-2/clusters/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

@0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Atotal
	Bcount
C	variables
D	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

C	variables*
��
VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_1/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_1/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE SGD/clustering/clusters/momentum[layer_with_weights-2/clusters/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:���������P*
dtype0*
shape:���������P
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasclustering/clusters*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_33763862
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp'clustering/clusters/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-SGD/dense/kernel/momentum/Read/ReadVariableOp+SGD/dense/bias/momentum/Read/ReadVariableOp/SGD/dense_1/kernel/momentum/Read/ReadVariableOp-SGD/dense_1/bias/momentum/Read/ReadVariableOp4SGD/clustering/clusters/momentum/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_33764052
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasclustering/clustersSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcountSGD/dense/kernel/momentumSGD/dense/bias/momentumSGD/dense_1/kernel/momentumSGD/dense_1/bias/momentum SGD/clustering/clusters/momentum*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_33764110��
�

�
E__inference_dense_1_layer_call_and_return_conditional_losses_33763448

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_clustering_layer_call_and_return_conditional_losses_33763970

inputs-
sub_readvariableop_resource:
identity��sub/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������n
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes

:*
dtype0q
subSubExpandDims:output:0sub/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������O
SquareSquaresub:z:0*
T0*+
_output_shapes
:���������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :h
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
addAddV2add/x:output:0truediv:z:0*
T0*'
_output_shapes
:���������P
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
	truediv_1RealDivtruediv_1/x:output:0add:z:0*
T0*'
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
powPowtruediv_1:z:0pow/y:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       j
	transpose	Transposepow:z:0transpose/perm:output:0*
T0*'
_output_shapes
:���������Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
Sum_1Sumpow:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:���������e
	truediv_2RealDivtranspose:y:0Sum_1:output:0*
T0*'
_output_shapes
:���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       t
transpose_1	Transposetruediv_2:z:0transpose_1/perm:output:0*
T0*'
_output_shapes
:���������^
IdentityIdentitytranspose_1:y:0^NoOp*
T0*'
_output_shapes
:���������[
NoOpNoOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2(
sub/ReadVariableOpsub/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
/__inference_dense_activity_regularizer_33763405
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�
�
*__inference_dense_1_layer_call_fn_33763891

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_33763448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
-__inference_clustering_layer_call_fn_33763916

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_clustering_layer_call_and_return_conditional_losses_33763538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_model_2_layer_call_fn_33763721

inputs
unknown:P@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������: *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_33763487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
H__inference_clustering_layer_call_and_return_conditional_losses_33763538

inputs-
sub_readvariableop_resource:
identity��sub/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������n
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes

:*
dtype0q
subSubExpandDims:output:0sub/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������O
SquareSquaresub:z:0*
T0*+
_output_shapes
:���������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :h
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
addAddV2add/x:output:0truediv:z:0*
T0*'
_output_shapes
:���������P
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
	truediv_1RealDivtruediv_1/x:output:0add:z:0*
T0*'
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
powPowtruediv_1:z:0pow/y:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       j
	transpose	Transposepow:z:0transpose/perm:output:0*
T0*'
_output_shapes
:���������Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
Sum_1Sumpow:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:���������e
	truediv_2RealDivtranspose:y:0Sum_1:output:0*
T0*'
_output_shapes
:���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       t
transpose_1	Transposetruediv_2:z:0transpose_1/perm:output:0*
T0*'
_output_shapes
:���������^
IdentityIdentitytranspose_1:y:0^NoOp*
T0*'
_output_shapes
:���������[
NoOpNoOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2(
sub/ReadVariableOpsub/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_dense_layer_call_and_return_all_conditional_losses_33763882

inputs
unknown:P@
	unknown_0:@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_33763423�
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *8
f3R1
/__inference_dense_activity_regularizer_33763405o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�B
�	
$__inference__traced_restore_33764110
file_prefix/
assignvariableop_dense_kernel:P@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@-
assignvariableop_3_dense_1_bias:8
&assignvariableop_4_clustering_clusters:%
assignvariableop_5_sgd_iter:	 &
assignvariableop_6_sgd_decay: .
$assignvariableop_7_sgd_learning_rate: )
assignvariableop_8_sgd_momentum: "
assignvariableop_9_total: #
assignvariableop_10_count: ?
-assignvariableop_11_sgd_dense_kernel_momentum:P@9
+assignvariableop_12_sgd_dense_bias_momentum:@A
/assignvariableop_13_sgd_dense_1_kernel_momentum:@;
-assignvariableop_14_sgd_dense_1_bias_momentum:F
4assignvariableop_15_sgd_clustering_clusters_momentum:
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/clusters/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/clusters/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_clustering_clustersIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_sgd_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp-assignvariableop_11_sgd_dense_kernel_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_sgd_dense_bias_momentumIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_sgd_dense_1_kernel_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_sgd_dense_1_bias_momentumIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp4assignvariableop_15_sgd_clustering_clusters_momentumIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
C__inference_dense_layer_call_and_return_conditional_losses_33763981

inputs0
matmul_readvariableop_resource:P@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�*
�
!__inference__traced_save_33764052
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop2
.savev2_clustering_clusters_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_sgd_dense_kernel_momentum_read_readvariableop6
2savev2_sgd_dense_bias_momentum_read_readvariableop:
6savev2_sgd_dense_1_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_1_bias_momentum_read_readvariableop?
;savev2_sgd_clustering_clusters_momentum_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/clusters/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/clusters/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop.savev2_clustering_clusters_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableop6savev2_sgd_dense_1_kernel_momentum_read_readvariableop4savev2_sgd_dense_1_bias_momentum_read_readvariableop;savev2_sgd_clustering_clusters_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*w
_input_shapesf
d: :P@:@:@::: : : : : : :P@:@:@::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:P@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:P@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: 
�
�
E__inference_model_2_layer_call_and_return_conditional_losses_33763701
input_1 
dense_33763678:P@
dense_33763680:@"
dense_1_33763691:@
dense_1_33763693:%
clustering_33763696:
identity

identity_1��"clustering/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_33763678dense_33763680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_33763423�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *8
f3R1
/__inference_dense_activity_regularizer_33763405u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33763691dense_1_33763693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_33763448�
"clustering/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0clustering_33763696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_clustering_layer_call_and_return_conditional_losses_33763538z
IdentityIdentity+clustering/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^clustering/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 2H
"clustering/StatefulPartitionedCall"clustering/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������P
!
_user_specified_name	input_1
�>
�
#__inference__wrapped_model_33763392
input_1>
,model_2_dense_matmul_readvariableop_resource:P@;
-model_2_dense_biasadd_readvariableop_resource:@@
.model_2_dense_1_matmul_readvariableop_resource:@=
/model_2_dense_1_biasadd_readvariableop_resource:@
.model_2_clustering_sub_readvariableop_resource:
identity��%model_2/clustering/sub/ReadVariableOp�$model_2/dense/BiasAdd/ReadVariableOp�#model_2/dense/MatMul/ReadVariableOp�&model_2/dense_1/BiasAdd/ReadVariableOp�%model_2/dense_1/MatMul/ReadVariableOp�
#model_2/dense/MatMul/ReadVariableOpReadVariableOp,model_2_dense_matmul_readvariableop_resource*
_output_shapes

:P@*
dtype0�
model_2/dense/MatMulMatMulinput_1+model_2/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model_2/dense/BiasAdd/ReadVariableOpReadVariableOp-model_2_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_2/dense/BiasAddBiasAddmodel_2/dense/MatMul:product:0,model_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model_2/dense/ReluRelumodel_2/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
%model_2/dense/ActivityRegularizer/AbsAbs model_2/dense/Relu:activations:0*
T0*'
_output_shapes
:���������@x
'model_2/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%model_2/dense/ActivityRegularizer/SumSum)model_2/dense/ActivityRegularizer/Abs:y:00model_2/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: l
'model_2/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%model_2/dense/ActivityRegularizer/mulMul0model_2/dense/ActivityRegularizer/mul/x:output:0.model_2/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: w
'model_2/dense/ActivityRegularizer/ShapeShape model_2/dense/Relu:activations:0*
T0*
_output_shapes
:
5model_2/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7model_2/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7model_2/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/model_2/dense/ActivityRegularizer/strided_sliceStridedSlice0model_2/dense/ActivityRegularizer/Shape:output:0>model_2/dense/ActivityRegularizer/strided_slice/stack:output:0@model_2/dense/ActivityRegularizer/strided_slice/stack_1:output:0@model_2/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
&model_2/dense/ActivityRegularizer/CastCast8model_2/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
)model_2/dense/ActivityRegularizer/truedivRealDiv)model_2/dense/ActivityRegularizer/mul:z:0*model_2/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
%model_2/dense_1/MatMul/ReadVariableOpReadVariableOp.model_2_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_2/dense_1/MatMulMatMul model_2/dense/Relu:activations:0-model_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_2/dense_1/BiasAddBiasAdd model_2/dense_1/MatMul:product:0.model_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
model_2/dense_1/ReluRelu model_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������c
!model_2/clustering/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/clustering/ExpandDims
ExpandDims"model_2/dense_1/Relu:activations:0*model_2/clustering/ExpandDims/dim:output:0*
T0*+
_output_shapes
:����������
%model_2/clustering/sub/ReadVariableOpReadVariableOp.model_2_clustering_sub_readvariableop_resource*
_output_shapes

:*
dtype0�
model_2/clustering/subSub&model_2/clustering/ExpandDims:output:0-model_2/clustering/sub/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������u
model_2/clustering/SquareSquaremodel_2/clustering/sub:z:0*
T0*+
_output_shapes
:���������j
(model_2/clustering/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/clustering/SumSummodel_2/clustering/Square:y:01model_2/clustering/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������a
model_2/clustering/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_2/clustering/truedivRealDivmodel_2/clustering/Sum:output:0%model_2/clustering/truediv/y:output:0*
T0*'
_output_shapes
:���������]
model_2/clustering/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_2/clustering/addAddV2!model_2/clustering/add/x:output:0model_2/clustering/truediv:z:0*
T0*'
_output_shapes
:���������c
model_2/clustering/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_2/clustering/truediv_1RealDiv'model_2/clustering/truediv_1/x:output:0model_2/clustering/add:z:0*
T0*'
_output_shapes
:���������]
model_2/clustering/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_2/clustering/powPow model_2/clustering/truediv_1:z:0!model_2/clustering/pow/y:output:0*
T0*'
_output_shapes
:���������r
!model_2/clustering/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
model_2/clustering/transpose	Transposemodel_2/clustering/pow:z:0*model_2/clustering/transpose/perm:output:0*
T0*'
_output_shapes
:���������l
*model_2/clustering/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/clustering/Sum_1Summodel_2/clustering/pow:z:03model_2/clustering/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:����������
model_2/clustering/truediv_2RealDiv model_2/clustering/transpose:y:0!model_2/clustering/Sum_1:output:0*
T0*'
_output_shapes
:���������t
#model_2/clustering/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
model_2/clustering/transpose_1	Transpose model_2/clustering/truediv_2:z:0,model_2/clustering/transpose_1/perm:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"model_2/clustering/transpose_1:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^model_2/clustering/sub/ReadVariableOp%^model_2/dense/BiasAdd/ReadVariableOp$^model_2/dense/MatMul/ReadVariableOp'^model_2/dense_1/BiasAdd/ReadVariableOp&^model_2/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 2N
%model_2/clustering/sub/ReadVariableOp%model_2/clustering/sub/ReadVariableOp2L
$model_2/dense/BiasAdd/ReadVariableOp$model_2/dense/BiasAdd/ReadVariableOp2J
#model_2/dense/MatMul/ReadVariableOp#model_2/dense/MatMul/ReadVariableOp2P
&model_2/dense_1/BiasAdd/ReadVariableOp&model_2/dense_1/BiasAdd/ReadVariableOp2N
%model_2/dense_1/MatMul/ReadVariableOp%model_2/dense_1/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������P
!
_user_specified_name	input_1
�
�
*__inference_model_2_layer_call_fn_33763501
input_1
unknown:P@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������: *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_33763487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������P
!
_user_specified_name	input_1
�
�
*__inference_model_2_layer_call_fn_33763649
input_1
unknown:P@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������: *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_33763619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������P
!
_user_specified_name	input_1
�
�
E__inference_model_2_layer_call_and_return_conditional_losses_33763619

inputs 
dense_33763596:P@
dense_33763598:@"
dense_1_33763609:@
dense_1_33763611:%
clustering_33763614:
identity

identity_1��"clustering/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_33763596dense_33763598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_33763423�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *8
f3R1
/__inference_dense_activity_regularizer_33763405u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33763609dense_1_33763611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_33763448�
"clustering/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0clustering_33763614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_clustering_layer_call_and_return_conditional_losses_33763538z
IdentityIdentity+clustering/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^clustering/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 2H
"clustering/StatefulPartitionedCall"clustering/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�8
�
E__inference_model_2_layer_call_and_return_conditional_losses_33763845

inputs6
$dense_matmul_readvariableop_resource:P@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&clustering_sub_readvariableop_resource:
identity

identity_1��clustering/sub/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:P@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@p
dense/ActivityRegularizer/AbsAbsdense/Relu:activations:0*
T0*'
_output_shapes
:���������@p
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: g
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������[
clustering/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
clustering/ExpandDims
ExpandDimsdense_1/Relu:activations:0"clustering/ExpandDims/dim:output:0*
T0*+
_output_shapes
:����������
clustering/sub/ReadVariableOpReadVariableOp&clustering_sub_readvariableop_resource*
_output_shapes

:*
dtype0�
clustering/subSubclustering/ExpandDims:output:0%clustering/sub/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������e
clustering/SquareSquareclustering/sub:z:0*
T0*+
_output_shapes
:���������b
 clustering/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
clustering/SumSumclustering/Square:y:0)clustering/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������Y
clustering/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clustering/truedivRealDivclustering/Sum:output:0clustering/truediv/y:output:0*
T0*'
_output_shapes
:���������U
clustering/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
clustering/addAddV2clustering/add/x:output:0clustering/truediv:z:0*
T0*'
_output_shapes
:���������[
clustering/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clustering/truediv_1RealDivclustering/truediv_1/x:output:0clustering/add:z:0*
T0*'
_output_shapes
:���������U
clustering/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
clustering/powPowclustering/truediv_1:z:0clustering/pow/y:output:0*
T0*'
_output_shapes
:���������j
clustering/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
clustering/transpose	Transposeclustering/pow:z:0"clustering/transpose/perm:output:0*
T0*'
_output_shapes
:���������d
"clustering/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
clustering/Sum_1Sumclustering/pow:z:0+clustering/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:����������
clustering/truediv_2RealDivclustering/transpose:y:0clustering/Sum_1:output:0*
T0*'
_output_shapes
:���������l
clustering/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
clustering/transpose_1	Transposeclustering/truediv_2:z:0$clustering/transpose_1/perm:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentityclustering/transpose_1:y:0^NoOp*
T0*'
_output_shapes
:���������e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^clustering/sub/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 2>
clustering/sub/ReadVariableOpclustering/sub/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
E__inference_model_2_layer_call_and_return_conditional_losses_33763675
input_1 
dense_33763652:P@
dense_33763654:@"
dense_1_33763665:@
dense_1_33763667:%
clustering_33763670:
identity

identity_1��"clustering/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_33763652dense_33763654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_33763423�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *8
f3R1
/__inference_dense_activity_regularizer_33763405u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33763665dense_1_33763667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_33763448�
"clustering/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0clustering_33763670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_clustering_layer_call_and_return_conditional_losses_33763481z
IdentityIdentity+clustering/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^clustering/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 2H
"clustering/StatefulPartitionedCall"clustering/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������P
!
_user_specified_name	input_1
�
�
&__inference_signature_wrapper_33763862
input_1
unknown:P@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_33763392o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������P
!
_user_specified_name	input_1
�
�
-__inference_clustering_layer_call_fn_33763909

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_clustering_layer_call_and_return_conditional_losses_33763481o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_model_2_layer_call_and_return_conditional_losses_33763487

inputs 
dense_33763424:P@
dense_33763426:@"
dense_1_33763449:@
dense_1_33763451:%
clustering_33763482:
identity

identity_1��"clustering/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_33763424dense_33763426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_33763423�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *8
f3R1
/__inference_dense_activity_regularizer_33763405u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33763449dense_1_33763451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_33763448�
"clustering/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0clustering_33763482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_clustering_layer_call_and_return_conditional_losses_33763481z
IdentityIdentity+clustering/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^clustering/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 2H
"clustering/StatefulPartitionedCall"clustering/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�

�
E__inference_dense_1_layer_call_and_return_conditional_losses_33763902

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_layer_call_and_return_conditional_losses_33763423

inputs0
matmul_readvariableop_resource:P@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�8
�
E__inference_model_2_layer_call_and_return_conditional_losses_33763791

inputs6
$dense_matmul_readvariableop_resource:P@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&clustering_sub_readvariableop_resource:
identity

identity_1��clustering/sub/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:P@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@p
dense/ActivityRegularizer/AbsAbsdense/Relu:activations:0*
T0*'
_output_shapes
:���������@p
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: g
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������[
clustering/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
clustering/ExpandDims
ExpandDimsdense_1/Relu:activations:0"clustering/ExpandDims/dim:output:0*
T0*+
_output_shapes
:����������
clustering/sub/ReadVariableOpReadVariableOp&clustering_sub_readvariableop_resource*
_output_shapes

:*
dtype0�
clustering/subSubclustering/ExpandDims:output:0%clustering/sub/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������e
clustering/SquareSquareclustering/sub:z:0*
T0*+
_output_shapes
:���������b
 clustering/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
clustering/SumSumclustering/Square:y:0)clustering/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������Y
clustering/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clustering/truedivRealDivclustering/Sum:output:0clustering/truediv/y:output:0*
T0*'
_output_shapes
:���������U
clustering/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
clustering/addAddV2clustering/add/x:output:0clustering/truediv:z:0*
T0*'
_output_shapes
:���������[
clustering/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clustering/truediv_1RealDivclustering/truediv_1/x:output:0clustering/add:z:0*
T0*'
_output_shapes
:���������U
clustering/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
clustering/powPowclustering/truediv_1:z:0clustering/pow/y:output:0*
T0*'
_output_shapes
:���������j
clustering/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
clustering/transpose	Transposeclustering/pow:z:0"clustering/transpose/perm:output:0*
T0*'
_output_shapes
:���������d
"clustering/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
clustering/Sum_1Sumclustering/pow:z:0+clustering/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:����������
clustering/truediv_2RealDivclustering/transpose:y:0clustering/Sum_1:output:0*
T0*'
_output_shapes
:���������l
clustering/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
clustering/transpose_1	Transposeclustering/truediv_2:z:0$clustering/transpose_1/perm:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentityclustering/transpose_1:y:0^NoOp*
T0*'
_output_shapes
:���������e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^clustering/sub/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 2>
clustering/sub/ReadVariableOpclustering/sub/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
H__inference_clustering_layer_call_and_return_conditional_losses_33763943

inputs-
sub_readvariableop_resource:
identity��sub/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������n
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes

:*
dtype0q
subSubExpandDims:output:0sub/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������O
SquareSquaresub:z:0*
T0*+
_output_shapes
:���������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :h
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
addAddV2add/x:output:0truediv:z:0*
T0*'
_output_shapes
:���������P
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
	truediv_1RealDivtruediv_1/x:output:0add:z:0*
T0*'
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
powPowtruediv_1:z:0pow/y:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       j
	transpose	Transposepow:z:0transpose/perm:output:0*
T0*'
_output_shapes
:���������Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
Sum_1Sumpow:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:���������e
	truediv_2RealDivtranspose:y:0Sum_1:output:0*
T0*'
_output_shapes
:���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       t
transpose_1	Transposetruediv_2:z:0transpose_1/perm:output:0*
T0*'
_output_shapes
:���������^
IdentityIdentitytranspose_1:y:0^NoOp*
T0*'
_output_shapes
:���������[
NoOpNoOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2(
sub/ReadVariableOpsub/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_layer_call_fn_33763871

inputs
unknown:P@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_33763423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
H__inference_clustering_layer_call_and_return_conditional_losses_33763481

inputs-
sub_readvariableop_resource:
identity��sub/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������n
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes

:*
dtype0q
subSubExpandDims:output:0sub/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������O
SquareSquaresub:z:0*
T0*+
_output_shapes
:���������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :h
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
truedivRealDivSum:output:0truediv/y:output:0*
T0*'
_output_shapes
:���������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
addAddV2add/x:output:0truediv:z:0*
T0*'
_output_shapes
:���������P
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
	truediv_1RealDivtruediv_1/x:output:0add:z:0*
T0*'
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
powPowtruediv_1:z:0pow/y:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       j
	transpose	Transposepow:z:0transpose/perm:output:0*
T0*'
_output_shapes
:���������Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
Sum_1Sumpow:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:���������e
	truediv_2RealDivtranspose:y:0Sum_1:output:0*
T0*'
_output_shapes
:���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       t
transpose_1	Transposetruediv_2:z:0transpose_1/perm:output:0*
T0*'
_output_shapes
:���������^
IdentityIdentitytranspose_1:y:0^NoOp*
T0*'
_output_shapes
:���������[
NoOpNoOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2(
sub/ReadVariableOpsub/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_model_2_layer_call_fn_33763737

inputs
unknown:P@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������: *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_33763619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������P: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������P>

clustering0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�T
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
clusters
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
�
%iter
	&decay
'learning_rate
(momentummomentumEmomentumFmomentumGmomentumHmomentumI"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_model_2_layer_call_fn_33763501
*__inference_model_2_layer_call_fn_33763721
*__inference_model_2_layer_call_fn_33763737
*__inference_model_2_layer_call_fn_33763649�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_model_2_layer_call_and_return_conditional_losses_33763791
E__inference_model_2_layer_call_and_return_conditional_losses_33763845
E__inference_model_2_layer_call_and_return_conditional_losses_33763675
E__inference_model_2_layer_call_and_return_conditional_losses_33763701�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference__wrapped_model_33763392input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
.serving_default"
signature_map
:P@2dense/kernel
:@2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__
4activity_regularizer_fn
*&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_layer_call_fn_33763871�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_dense_layer_call_and_return_all_conditional_losses_33763882�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 :@2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_1_layer_call_fn_33763891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_1_layer_call_and_return_conditional_losses_33763902�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
%:#2clustering/clusters
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_clustering_layer_call_fn_33763909
-__inference_clustering_layer_call_fn_33763916�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
H__inference_clustering_layer_call_and_return_conditional_losses_33763943
H__inference_clustering_layer_call_and_return_conditional_losses_33763970�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_signature_wrapper_33763862input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
/__inference_dense_activity_regularizer_33763405�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�2�
C__inference_dense_layer_call_and_return_conditional_losses_33763981�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Atotal
	Bcount
C	variables
D	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
):'P@2SGD/dense/kernel/momentum
#:!@2SGD/dense/bias/momentum
+:)@2SGD/dense_1/kernel/momentum
%:#2SGD/dense_1/bias/momentum
0:.2 SGD/clustering/clusters/momentum�
#__inference__wrapped_model_33763392r0�-
&�#
!�
input_1���������P
� "7�4
2

clustering$�!

clustering����������
H__inference_clustering_layer_call_and_return_conditional_losses_33763943k?�<
%�"
 �
inputs���������
�

trainingp "%�"
�
0���������
� �
H__inference_clustering_layer_call_and_return_conditional_losses_33763970k?�<
%�"
 �
inputs���������
�

trainingp"%�"
�
0���������
� �
-__inference_clustering_layer_call_fn_33763909^?�<
%�"
 �
inputs���������
�

trainingp "�����������
-__inference_clustering_layer_call_fn_33763916^?�<
%�"
 �
inputs���������
�

trainingp"�����������
E__inference_dense_1_layer_call_and_return_conditional_losses_33763902\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_1_layer_call_fn_33763891O/�,
%�"
 �
inputs���������@
� "����������Y
/__inference_dense_activity_regularizer_33763405&�
�
�	
x
� "� �
G__inference_dense_layer_call_and_return_all_conditional_losses_33763882j/�,
%�"
 �
inputs���������P
� "3�0
�
0���������@
�
�	
1/0 �
C__inference_dense_layer_call_and_return_conditional_losses_33763981\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������@
� {
(__inference_dense_layer_call_fn_33763871O/�,
%�"
 �
inputs���������P
� "����������@�
E__inference_model_2_layer_call_and_return_conditional_losses_33763675v8�5
.�+
!�
input_1���������P
p 

 
� "3�0
�
0���������
�
�	
1/0 �
E__inference_model_2_layer_call_and_return_conditional_losses_33763701v8�5
.�+
!�
input_1���������P
p

 
� "3�0
�
0���������
�
�	
1/0 �
E__inference_model_2_layer_call_and_return_conditional_losses_33763791u7�4
-�*
 �
inputs���������P
p 

 
� "3�0
�
0���������
�
�	
1/0 �
E__inference_model_2_layer_call_and_return_conditional_losses_33763845u7�4
-�*
 �
inputs���������P
p

 
� "3�0
�
0���������
�
�	
1/0 �
*__inference_model_2_layer_call_fn_33763501[8�5
.�+
!�
input_1���������P
p 

 
� "�����������
*__inference_model_2_layer_call_fn_33763649[8�5
.�+
!�
input_1���������P
p

 
� "�����������
*__inference_model_2_layer_call_fn_33763721Z7�4
-�*
 �
inputs���������P
p 

 
� "�����������
*__inference_model_2_layer_call_fn_33763737Z7�4
-�*
 �
inputs���������P
p

 
� "�����������
&__inference_signature_wrapper_33763862};�8
� 
1�.
,
input_1!�
input_1���������P"7�4
2

clustering$�!

clustering���������