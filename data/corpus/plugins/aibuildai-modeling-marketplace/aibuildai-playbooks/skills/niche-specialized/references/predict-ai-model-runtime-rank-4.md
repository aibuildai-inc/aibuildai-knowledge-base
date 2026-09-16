# 4th simple mlp solution

Competition: predict-ai-model-runtime
Rank: #4
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456462

`class SimpleMLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(120,10)
        self.config_dense = torch.nn.Sequential(nn.Linear(128+8, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 128),
                                                nn.ReLU(),
                                        )
        self.node_dense = torch.nn.Sequential(nn.Linear(12*2+20, 1024),
                                              nn.ReLU(),
                                              nn.Linear(1024, 128),
                                              nn.ReLU(),
                                        )
        self.output=torch.nn.Sequential(nn.Linear(128*128+128, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1),
                                        )
    def forward(self,x_cfg,x_feat,x_op):
        x_feat=x_feat.squeeze()
        x_op=x_op.squeeze()
        x_op = self.embedding(x_op).reshape(-1,20)
        x_feat = torch.concat([x_feat,x_op],dim =1)
        x_feat = self.node_dense(x_feat)
        x_graph = x_feat.unsqueeze(0)
        x_graph = x_graph.repeat_interleave(len(x_cfg),dim=0)
        x_cfg = torch.concat([x_cfg,x_graph],axis=2)
        x_cfg = self.config_dense(x_cfg)
        x=(x_feat.T@x_cfg).reshape(len(x_cfg),-1)
        x_cfg_mean=x_cfg.mean(dim=1)
        x = torch.concat([x,x_cfg_mean],axis=1)
        x=self.output(x)
        x=torch.flatten(x)
        return x
    

def load_data(row):

    node_feat_index=[21,22,23,24,28,
                 101,102,103,104,
                 134,135,136,
                ]
    config_index=[0,1,2,6,7,8,12,13]
    data= dict(np.load(row.path))
    X=data["node_feat"][:,node_feat_index]
    node_feat=pd.DataFrame(X,columns=["fe%s"% i for i in range(X.shape[1])])
    node_feat["id"]=range(len(node_feat))
    node_feat_link=node_feat.copy()
    node_feat_link.columns=[i+"_link" for i in node_feat_link.columns]
    node_config_ids=pd.DataFrame({"id":data["node_config_ids"],"ind":range(len(data["node_config_ids"]))})
    node_opcode=pd.DataFrame({"id":range(len(data["node_opcode"])),"node_opcode":data["node_opcode"]})
    edge_index=pd.DataFrame(data["edge_index"],columns=["id","id_link"])
    edge_index=edge_index[edge_index.id.isin(data["node_config_ids"])]
    edge_index=edge_index.merge(node_opcode,on="id",how="left").merge(node_opcode.rename(columns={"id":"id_link","node_opcode":"node_opcode_link"}),on="id_link",how="left")
    edge_index=edge_index.merge(node_config_ids,on="id",how="left")
    edge_index=edge_index.merge(node_feat,on="id",how="left").merge(node_feat_link,on="id_link",how="left")
    all_features=[i for i in edge_index.columns if i not in ['id', 'id_link', 'node_opcode','node_opcode_link', 'ind']]
    node_feat_array=edge_index[all_features].values.astype(np.float32)
    node_config_feat=data["node_config_feat"][:,:,config_index]
    node_opcode_array=edge_index[["node_opcode","node_opcode_link"]].values
    node_config_feat=node_config_feat[:,edge_index["ind"].values,:]
    label=data["config_runtime"].argsort().argsort()/len(data["config_runtime"])
    return {"node_feat":node_feat_array,              #x_feat
            "node_config_feat":node_config_feat,      #x_cfg
            "node_opcode":node_opcode_array,          #x_op
            "target":label,
            "config_runtime":data["config_runtime"],
            "ind":np.array(range(len(label)))
           }`
