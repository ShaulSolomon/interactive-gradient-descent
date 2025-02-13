import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(layout="wide")

# Initialize session state variables
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'weight' not in st.session_state:
    st.session_state.weight = None
if 'bias' not in st.session_state:
    st.session_state.bias = None
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'weight_history' not in st.session_state:
    st.session_state.weight_history = []
if 'bias_history' not in st.session_state:
    st.session_state.bias_history = []
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'true_weight' not in st.session_state:
    st.session_state.true_weight = None
if 'true_bias' not in st.session_state:
    st.session_state.true_bias = None
if 'last_weight_val' not in st.session_state:
    st.session_state.last_weight_val = None
if 'last_bias_val' not in st.session_state:
    st.session_state.last_bias_val = None
if 'last_loss_val' not in st.session_state:
    st.session_state.last_loss_val = None

def compute_loss(X, y, weight, bias, loss_type="MSE"):
    predictions = X * weight + bias
    if loss_type == "MSE":
        return np.mean((predictions - y) ** 2)
    elif loss_type == "MAE":
        return np.mean(np.abs(predictions - y))
    else:  # RMSE
        return np.sqrt(np.mean((predictions - y) ** 2))

def compute_gradients(X, y, weight, bias, loss_type="MSE"):
    predictions = X * weight + bias
    errors = predictions - y
    
    if loss_type == "MSE":
        dw = 2 * np.mean(X * errors)
        db = 2 * np.mean(errors)
    elif loss_type == "MAE":
        dw = np.mean(X * np.sign(errors))
        db = np.mean(np.sign(errors))
    else:  # RMSE
        mse = np.mean(errors ** 2)
        dw = np.mean(X * errors) / np.sqrt(mse)
        db = np.mean(errors) / np.sqrt(mse)
    return dw, db

def compute_loss_curve(X, y, param_range, fixed_param, param_type='weight', loss_type="MSE"):
    losses = []
    for param in param_range:
        if param_type == 'weight':
            loss = compute_loss(X, y, param, fixed_param, loss_type)
        else:
            loss = compute_loss(X, y, fixed_param, param, loss_type)
        losses.append(loss)
    return np.array(losses)

def generate_medical_data(n_samples=100, noise=0.3):
    """Generate synthetic medical data: Treatment Duration vs Recovery Time"""
    X = np.random.uniform(0, 10, (n_samples, 1))
    
    # Generate random true parameters in reasonable ranges
    true_weight = np.random.uniform(-7.0, 7.0)  # Recovery rate between 1.5 and 3.5 days per treatment unit
    true_bias = np.random.uniform(-35, 35)      # Base recovery time between 25 and 35 days
    
    y = X * true_weight + true_bias + noise * np.random.randn(n_samples, 1)
    return X, y, true_weight, true_bias

def generate_data():
    # Generate data with random true parameters
    X, y, true_weight, true_bias = generate_medical_data(
        st.session_state.n_samples,
        st.session_state.noise
    )
    
    # Store everything in session state
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.true_weight = true_weight
    st.session_state.true_bias = true_bias
    st.session_state.data_generated = True
    
    # Reset optimization state
    st.session_state.weight = None
    st.session_state.bias = None
    st.session_state.loss_history = []
    st.session_state.weight_history = []
    st.session_state.bias_history = []

def initialize_parameters():

    st.session_state.weight = np.random.uniform(-10.0, 10.0)
    st.session_state.bias =np.random.uniform(-50, 50)  
    st.session_state.loss_history = []
    st.session_state.weight_history = []
    st.session_state.bias_history = []

    # Calculate initial loss immediately
    initial_loss = compute_loss(
        st.session_state.X,
        st.session_state.y,
        st.session_state.weight,
        st.session_state.bias,
        st.session_state.loss_type
    )
    st.session_state.loss_history.append(initial_loss)
    st.session_state.weight_history.append(st.session_state.weight)
    st.session_state.bias_history.append(st.session_state.bias)

def take_gradient_step():
    dw, db = compute_gradients(
        st.session_state.X,
        st.session_state.y,
        st.session_state.weight,
        st.session_state.bias,
        st.session_state.loss_type
    )
    
    st.session_state.weight -= st.session_state.learning_rate * dw
    st.session_state.bias -= st.session_state.learning_rate * db
    
    loss = compute_loss(
        st.session_state.X,
        st.session_state.y,
        st.session_state.weight,
        st.session_state.bias,
        st.session_state.loss_type
    )
    
    st.session_state.loss_history.append(loss)
    st.session_state.weight_history.append(st.session_state.weight)
    st.session_state.bias_history.append(st.session_state.bias)

def get_parameter_changes():
    """Calculate changes in parameters from the previous step"""
    if len(st.session_state.weight_history) < 2:
        return None, None, None

    # Get changes from previous step
    weight_change = st.session_state.weight_history[-1] - st.session_state.last_weight_val
    bias_change = st.session_state.bias_history[-1] - st.session_state.last_bias_val
    loss_change = st.session_state.loss_history[-1] - st.session_state.last_loss_val

    return weight_change, bias_change, loss_change

def get_max_loss():
        """Calculate a reasonable maximum loss value based on the data scale"""
        y_range = st.session_state.y.max() - st.session_state.y.min()
        # For MSE/RMSE, a very poor prediction could be off by the full range
        if st.session_state.loss_type == "MSE":
            return (y_range ** 2) * 2  # Multiple by 2 for safety margin
        elif st.session_state.loss_type == "RMSE":
            return y_range * 1.5  # Sqrt makes the range smaller
        else:  # MAE
            return y_range * 2  # Maximum error would be the full range

st.title("Interactive Gradient Descent in Linear/Logistic Regression")

def take_multiple_steps(n_steps):
    for _ in range(n_steps):
        take_gradient_step()

# Sidebar controls
with st.sidebar:
    st.session_state.loss_type = st.selectbox(
        "Loss Function",
        ["MSE", "MAE", "RMSE"]
    )
    
    st.session_state.n_samples = st.slider("Sample Size", 50, 200, 100)
    st.session_state.learning_rate = st.slider(
        "Learning Rate", 0.001, 0.1, 0.01, format="%.3f"
    )
    st.session_state.noise = st.slider("Noise Level", 0.1, 1.0, 0.3, format="%.2f")
    
    if st.button("Generate New Data"):
        generate_data()
    
    if st.session_state.data_generated:
        if st.button("Initialize Parameters"):
            initialize_parameters()
        
        if st.session_state.weight is not None:
            col1, col2 = st.columns(2)
            st.session_state.last_weight_val = st.session_state.weight_history[-1]
            st.session_state.last_bias_val = st.session_state.bias_history[-1]
            st.session_state.last_loss_val = st.session_state.loss_history[-1]
            with col1:
                if st.button("Take Step"):
                    take_gradient_step()
            with col2:
                if st.button("100 Steps"):
                    take_multiple_steps(100)

# Main content
if st.session_state.data_generated:
    # First create the main data visualization plot
    fig_data = go.Figure()
    
    # Plot data points
    fig_data.add_trace(
        go.Scatter(
            x=st.session_state.X.flatten(),
            y=st.session_state.y.flatten(),
            mode='markers',
            name='Data Points',
            marker=dict(
                color='#2E91E5',
                size=8,
                opacity=0.6
            ),
            showlegend=False
        )
    )
    
    # Add regression line if parameters are initialized
    if st.session_state.weight is not None:
        x_range = np.linspace(st.session_state.X.min(), st.session_state.X.max(), 100)
        y_pred = x_range * st.session_state.weight + st.session_state.bias
        
        fig_data.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name='Current Fit',
                line=dict(
                    color='#FF4B4B',
                    width=2
                ),
                showlegend=False
            )
        )
    
    # Update layout for data visualization with dark theme
    fig_data.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        paper_bgcolor='rgb(17,17,17)',
        plot_bgcolor='rgb(17,17,17)',
        font=dict(color='white'),
        xaxis_title="Treatment Duration",
        yaxis_title="Recovery Time"
    )
    
    # Update axes with dark theme
    fig_data.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        color='white'
    )
    
    fig_data.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        color='white'
    )
    
    # Display the main data visualization
    st.plotly_chart(fig_data, use_container_width=True)

    # Create subplot for loss landscapes and history
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Weight (w₁) Loss Landscape",
            "Bias (w₀) Loss Landscape",
            f"{st.session_state.loss_type} Loss History"
        ),
        horizontal_spacing=0.1
    )

    weight_range = np.linspace(-50, 50, 400)  # Fixed range for weight
    bias_range = np.linspace(-100, 100, 400)    # Broader range with more points

    # Calculate broader ranges for weight and bias
    if st.session_state.weight is not None:
        current_loss = st.session_state.loss_history[-1]
    else:
        current_loss = None

    
    # Weight loss landscape
    weight_losses = compute_loss_curve(
        st.session_state.X,
        st.session_state.y,
        weight_range,
        st.session_state.bias if st.session_state.bias is not None else 0,
        'weight',
        st.session_state.loss_type
    )
    
    fig.add_trace(
        go.Scatter(
            x=weight_range,
            y=weight_losses,
            mode='lines',
            line=dict(color='#2E91E5', width=2),
            showlegend=False
        ),
        row=1, col=1
    )
    
    if st.session_state.weight is not None:
        fig.add_trace(
            go.Scatter(
                x=[st.session_state.weight],
                y=[current_loss],
                mode='markers',
                marker=dict(color='#FF4B4B', size=10),
                showlegend=False
            ),
            row=1, col=1
        )
    # Set view range for weight plot
    if st.session_state.weight is not None:
        view_center = st.session_state.weight
        fig.update_xaxes(
            range=[max(-50, view_center - 10), min(50, view_center + 10)],
            row=1, col=1
        )
    
    # Bias loss landscape
    bias_losses = compute_loss_curve(
        st.session_state.X,
        st.session_state.y,
        bias_range,
        st.session_state.weight if st.session_state.weight is not None else 0,
        'bias',
        st.session_state.loss_type
    )
    
    fig.add_trace(
        go.Scatter(
            x=bias_range,
            y=bias_losses,
            mode='lines',
            line=dict(color='#2E91E5', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    if st.session_state.bias is not None:
        fig.add_trace(
            go.Scatter(
                x=[st.session_state.bias],
                y=[current_loss],
                mode='markers',
                marker=dict(color='#FF4B4B', size=10),
                showlegend=False
            ),
            row=1, col=2
        )

    # Set view range for bias plot
    if st.session_state.bias is not None:
        view_center = st.session_state.bias
        fig.update_xaxes(
            range=[max(-100, view_center - 20), min(100, view_center + 20)],
            row=1, col=2
        )
    
    # Loss history
    if st.session_state.loss_history:
        fig.add_trace(
            go.Scatter(
                y=st.session_state.loss_history,
                mode='lines',
                line=dict(color='#2E91E5', width=2),
                showlegend=False
            ),
            row=1, col=3
        )
        max_historical_loss = max(st.session_state.loss_history)
        fig.update_yaxes(
            range=[0, max(max_historical_loss * 1.1, 100)],
            row=1, col=3
        )
    
    # Update layout with dark theme
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        paper_bgcolor='rgb(17,17,17)',
        plot_bgcolor='rgb(17,17,17)',
        font=dict(color='white')
    )
    
    # Update subplot titles with more space
    fig.update_annotations(y=1.1, font=dict(color='white'))
    
    # Update axes with dark theme
    for i in range(1, 4):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=1, col=i,
            color='white'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=1, col=i,
            color='white'
        )
    
    # Update axis labels
    fig.update_xaxes(title_text="Weight (w₁)", row=1, col=1)
    fig.update_xaxes(title_text="Bias (w₀)", row=1, col=2)
    fig.update_xaxes(title_text="Step", row=1, col=3)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=3)
    
    # Display the plots
    st.plotly_chart(fig, use_container_width=True)

if st.session_state.weight is not None and st.session_state.loss_history:
    # Display metrics in a prominent way
    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
            background-color: rgba(28, 28, 28, 0.9);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 10px 15px;
            border-radius: 5px;
            color: white;
            width: 100%;
        }
        
        div[data-testid="metric-container"] label {
            color: rgb(180, 180, 180);
        }
        
        div[data-testid="stHorizontalBlock"] {
            gap: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    # Calculate changes
    weight_change, bias_change, loss_change = get_parameter_changes()
    
    st.markdown("### Current Optimization State")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Loss",
            f"{st.session_state.loss_history[-1]:.4f}",
            f"{loss_change:+.4f}" if loss_change is not None else None
        )
    with col2:
        st.metric(
            "Current Weight (w₁)",
            f"{st.session_state.weight:.4f}",
            f"{weight_change:+.4f}" if weight_change is not None else None
        )
    with col3:
        st.metric(
            "Current Bias (w₀)",
            f"{st.session_state.bias:.4f}",
            f"{bias_change:+.4f}" if bias_change is not None else None
        )
        
    st.markdown("### True Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Added Noise",
            f"{st.session_state.noise:.3f}"
        )
    with col2:
        st.metric(
            "True Weight",
            f"{st.session_state.true_weight:.4f}"
        )
    with col3:
        st.metric(
            "True Bias",
            f"{st.session_state.true_bias:.4f}"
        )