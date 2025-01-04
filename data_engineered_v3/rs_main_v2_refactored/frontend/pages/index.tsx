import { useState } from 'react';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';

interface Recommendation {
  music: string;
  artist_name: string;
  genre: string;
  predicted_plays: number;
}

export default function Home() {
  const [formData, setFormData] = useState({
    user_id: '',
    age: '',
    gender: '',
    genre: '',
    music: '',
  });
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5000/api/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }

      const data = await response.json();
      if (data.status === 'success') {
        setRecommendations(data.recommendations);
      } else {
        throw new Error(data.message || 'Failed to get recommendations');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | { name?: string; value: unknown }>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name as string]: value,
    }));
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Music Recommendation System
        </Typography>

        <Paper sx={{ p: 3, mb: 3 }}>
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="User ID"
              name="user_id"
              value={formData.user_id}
              onChange={handleChange}
              margin="normal"
              required
            />
            <TextField
              fullWidth
              label="Age"
              name="age"
              type="number"
              value={formData.age}
              onChange={handleChange}
              margin="normal"
              required
            />
            <FormControl fullWidth margin="normal">
              <InputLabel>Gender</InputLabel>
              <Select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                required
              >
                <MenuItem value="M">Male</MenuItem>
                <MenuItem value="F">Female</MenuItem>
              </Select>
            </FormControl>
            <TextField
              fullWidth
              label="Genre"
              name="genre"
              value={formData.genre}
              onChange={handleChange}
              margin="normal"
              required
            />
            <TextField
              fullWidth
              label="Music"
              name="music"
              value={formData.music}
              onChange={handleChange}
              margin="normal"
              required
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              sx={{ mt: 2 }}
              disabled={loading}
            >
              {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
            </Button>
          </form>
        </Paper>

        {error && (
          <Typography color="error" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {recommendations.length > 0 && (
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Music</TableCell>
                  <TableCell>Artist</TableCell>
                  <TableCell>Genre</TableCell>
                  <TableCell>Predicted Plays</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {recommendations.map((rec, index) => (
                  <TableRow key={index}>
                    <TableCell>{rec.music}</TableCell>
                    <TableCell>{rec.artist_name}</TableCell>
                    <TableCell>{rec.genre}</TableCell>
                    <TableCell>{Math.round(rec.predicted_plays)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Box>
    </Container>
  );
}
