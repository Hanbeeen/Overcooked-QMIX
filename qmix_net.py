import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    """
    QMIX 에이전트 네트워크 (DRQN).
    
    구조:
        Input -> FC1 -> GRU -> FC2 -> Q-Values
    
    특징:
        - 시계열 데이터(Sequence) 처리를 위해 GRU(Gated Recurrent Unit) 사용.
        - 이전 행동(Last Action)과 에이전트 ID를 입력으로 함께 받아 식별력 강화.
    """
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNNAgent, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # 1. Feature Extractor (기초 특징 추출)
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        
        # 2. Recurrent Layer (기억 메모리)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        
        # 3. Action Head (행동별 Q값 출력)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

        # 초기화: 직교 초기화(Orthogonal Init)로 학습 안정성 확보
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self):
        """GRU의 은닉 상태(Hidden State)를 0으로 초기화"""
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        순전파 (Forward Pass)
        
        Args:
            inputs (Tensor): [Batch, Input_Dim]
            hidden_state (Tensor): [Batch, Hidden_Dim]
            
        Returns:
            q_values (Tensor): [Batch, N_Actions] - 각 행동의 가치
            new_hidden (Tensor): [Batch, Hidden_Dim] - 업데이트된 기억
        """
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        return q, h
    
    @property
    def trainable_params(self):
        """학습 가능한 파라미터 반환"""
        return self.parameters()


class HyperMixer(nn.Module):
    """
    QMIX의 핵심인 Mixing Network.
    
    역할:
        개별 에이전트의 Q값($Q_a$)들을 전역 상태($State$)를 고려하여
        하나의 통합된 값 $Q_{tot}$로 합칩니다.
    
    핵심 제약조건 (Monotonicity):
        $\frac{\partial Q_{tot}}{\partial Q_a} \ge 0$
        개인의 이득이 커지면 팀 전체의 이득도 커져야 합니다.
        이를 위해 가중치(Hypernet output)에 절대값(abs)을 취해 항상 양수로 만듭니다.
    """
    def __init__(self, n_agents, state_shape, mixing_embed_dim=32):
        super(HyperMixer, self).__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim

        # 1. Hypernetwork for Weights (Layer 1)
        # State를 입력받아 개별 Q값들의 가중치(w1)를 생성
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_shape, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents * mixing_embed_dim)
        )
        # Bias for Layer 1
        self.hyper_b1 = nn.Linear(state_shape, mixing_embed_dim)

        # 2. Hypernetwork for Weights (Layer 2)
        # Hidden Layer -> Output의 가중치(w2) 생성
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_shape, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )
        # Bias for Output (State-Value Function $V(s)$)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_shape, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        Args:
            agent_qs (Tensor): [Batch, Time, N_Agents] - 개별 Q값들
            states (Tensor): [Batch, Time, State_Dim] - 전역 상태 정보
        
        Returns:
            q_tot (Tensor): [Batch, Time, 1] - 통합된 팀 Q값
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_shape) # [Batch*Time, State_Dim]
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents) # [Batch*Time, 1, N_Agents]

        # 1. Layer 1 Weights & Bias 생성
        w1 = torch.abs(self.hyper_w1(states)) # 절대값 -> 양수 보장 (Monotonicity)
        w1 = w1.view(-1, self.n_agents, self.embed_dim) 
        b1 = self.hyper_b1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        
        # Multiply: q * w1
        # [BS, 1, N] @ [BS, N, Embed] -> [BS, 1, Embed]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # 2. Second Layer mixing
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        
        w2 = w2.view(-1, self.embed_dim, 1) # [BS, Embed, 1]
        b2 = b2.view(-1, 1, 1)
        
        # [BS, 1, Embed] @ [BS, Embed, 1] -> [BS, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(bs, -1, 1)
