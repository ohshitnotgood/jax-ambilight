import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import org.jetbrains.compose.resources.painterResource
import org.jetbrains.compose.ui.tooling.preview.Preview

import jax_ambilight.composeapp.generated.resources.Res
import jax_ambilight.composeapp.generated.resources.compose_multiplatform

@Composable
@Preview
fun App() {
    MaterialTheme {
        var showContent by remember { mutableStateOf(false) }
        Row() {
            Column(
                modifier = Modifier
                    .fillMaxWidth(0.3f)
            ) {
                Text("Select monitor", modifier = Modifier.padding(10.dp))
            }
            Column(
                Modifier
                    .fillMaxWidth()
            ) {
                Text("Hello World")
            }
        }
    }
}